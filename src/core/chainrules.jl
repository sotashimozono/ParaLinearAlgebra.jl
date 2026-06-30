# core/chainrules.jl — class-agnostic AD rrules for the ParaMatrix calculus.
# Each operation is linear or bilinear in the coefficients, so every pullback is
# exact: a scalar loss assembled from these backprops to every coefficient block.
# (The para-adjoint rrule is Laurent-specific and lives in classes/laurent.jl.)

# evaluate is LINEAR in the coefficients: ∂coeffs_k = conj(basis_k(p))·ȳ
function ChainRulesCore.rrule(::typeof(evaluate), A::ParaMatrix, p)
    w = basis(A.class, p)
    y = sum(w[i] * A.coeffs[i] for i in eachindex(w))
    function evaluate_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        c̄ = [conj(w[i]) * Ȳ for i in eachindex(w)]
        return (NoTangent(), Tangent{typeof(A)}(; coeffs=c̄, class=NoTangent()), NoTangent())
    end
    return y, evaluate_pullback
end

_ctc(x) = unthunk(x).coeffs   # output cotangent → coefficient vector

function ChainRulesCore.rrule(::typeof(+), A::ParaMatrix, B::ParaMatrix)
    # give A and B INDEPENDENT cotangent buffers: a mutation-based AD backend
    # (e.g. Mooncake) accumulating in place must not alias one into the other.
    plus_back(Ȳ) = (
        c̄=_ctc(Ȳ);
        (
            NoTangent(),
            Tangent{typeof(A)}(; coeffs=c̄, class=NoTangent()),
            Tangent{typeof(B)}(; coeffs=[copy(c) for c in c̄], class=NoTangent()),
        )
    )
    return A + B, plus_back
end

function ChainRulesCore.rrule(::typeof(-), A::ParaMatrix, B::ParaMatrix)
    minus_back(Ȳ) = (
        c̄=_ctc(Ȳ);
        (
            NoTangent(),
            Tangent{typeof(A)}(; coeffs=c̄, class=NoTangent()),
            Tangent{typeof(B)}(; coeffs=[-c for c in c̄], class=NoTangent()),
        )
    )
    return A - B, minus_back
end

function ChainRulesCore.rrule(::typeof(*), α::Number, A::ParaMatrix)
    function scal_back(Ȳ)
        c̄ = _ctc(Ȳ)
        ᾱ = sum(dot(A.coeffs[i], c̄[i]) for i in eachindex(c̄))
        return (
            NoTangent(),
            ᾱ,
            Tangent{typeof(A)}(; coeffs=[conj(α) * c for c in c̄], class=NoTangent()),
        )
    end
    return α * A, scal_back
end

function ChainRulesCore.rrule(::typeof(*), A::ParaMatrix, B::ParaMatrix)
    Y = A * B
    pA, pB, pC = powers(A.class), powers(B.class), powers(Y.class)
    pos = Dict(pC[i] => i for i in eachindex(pC))
    function times_back(Ȳ)
        oc = _ctc(Ȳ)
        Ā = [
            sum(oc[pos[pA[i] + pB[j]]] * B.coeffs[j]' for j in eachindex(pB)) for
            i in eachindex(pA)
        ]
        B̄ = [
            sum(A.coeffs[i]' * oc[pos[pA[i] + pB[j]]] for i in eachindex(pA)) for
            j in eachindex(pB)
        ]
        return (
            NoTangent(),
            Tangent{typeof(A)}(; coeffs=Ā, class=NoTangent()),
            Tangent{typeof(B)}(; coeffs=B̄, class=NoTangent()),
        )
    end
    return Y, times_back
end

function ChainRulesCore.rrule(::typeof(kron), A::ParaMatrix, B::ParaMatrix)
    Y = kron(A, B)
    pA, pB, pC = powers(A.class), powers(B.class), powers(Y.class)
    pos = Dict(pC[i] => i for i in eachindex(pC))
    dA1, dA2 = size(A)
    dB1, dB2 = size(B)
    function kron_back(Ȳ)
        oc = _ctc(Ȳ)
        Ā = [zeros(eltype(A.coeffs[1]), dA1, dA2) for _ in pA]
        B̄ = [zeros(eltype(B.coeffs[1]), dB1, dB2) for _ in pB]
        for i in eachindex(pA), j in eachindex(pB)
            R = oc[pos[pA[i] + pB[j]]]
            Ai = A.coeffs[i]
            Bj = B.coeffs[j]
            for a in 1:dA1, b in 1:dA2
                blk = @view R[((a - 1) * dB1 + 1):(a * dB1), ((b - 1) * dB2 + 1):(b * dB2)]
                Ā[i][a, b] += sum(blk .* conj(Bj))
                @. B̄[j] += conj(Ai[a, b]) * blk
            end
        end
        return (
            NoTangent(),
            Tangent{typeof(A)}(; coeffs=Ā, class=NoTangent()),
            Tangent{typeof(B)}(; coeffs=B̄, class=NoTangent()),
        )
    end
    return Y, kron_back
end
