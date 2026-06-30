# classes/laurent.jl — complex Laurent class z^k on the torus (the RING class),
# plus the algebra that is specific to the Laurent power-window structure:
# para-adjoint, paraeye/one, predicates, det, inv. The para-adjoint and these
# notions are meaningless for a generic class, so they live here, not in core/.

"""
    Laurent(lo, hi) <: FunctionClass
    Laurent(L)  = Laurent(-L, L)     # symmetric window (para-Hermitian objects)
    Analytic(L) = Laurent(0, L)      # one-sided (causal / outer factors)

Complex Laurent series in the angle `θ ∈ R/Z`, weights `exp(2πi k θ)` for
`k = lo:hi`. The ring class for ParaMatrix products and the para-adjoint;
the natural class for twisted-boundary / flux parameters.
"""
struct Laurent <: RingClass
    lo::Int
    hi::Int
    function Laurent(lo::Int, hi::Int)
        lo ≤ hi || throw(ArgumentError("Laurent window needs lo ≤ hi; got $lo:$hi"))
        return new(lo, hi)
    end
end
Laurent(L::Int) = Laurent(-L, L)

"""
    Analytic(L) = Laurent(0, L)

One-sided (causal / outer) Laurent window `0:L` — the class of analytic outer
factors (e.g. the result of [`spectral_factor`](@ref)).
"""
Analytic(L::Int) = Laurent(0, L)

powers(c::Laurent) = (c.lo):(c.hi)

# `cispi(x) = exp(iπx)` avoids 2π argument-reduction error (Float64-ulp accurate,
# BigFloat-safe). Element type follows `p` so Complex{BigFloat} flows through.
basis(c::Laurent, p) = [cispi(2 * k * p) for k in powers(c)]
function basis_deriv(c::Laurent, p)
    τ = oftype(real(float(p)), π)
    return [im * 2 * k * τ * cispi(2 * k * p) for k in powers(c)]
end

_prodclass(a::Laurent, b::Laurent) = Laurent(a.lo + b.lo, a.hi + b.hi)

# {exp(2πikθ)} is L²-orthonormal on [0,1) ⇒ the Gram is the identity (exact Parseval).
basis_gram(c::Laurent) = Matrix{Float64}(I, nbasis(c), nbasis(c))

# ∫₀¹ e^{2πikθ} dθ = δ_{k,0}: only the k=0 coefficient survives
basis_integral(c::Laurent) = [k == 0 ? 1.0 : 0.0 for k in powers(c)]

# `paraeye`/`one` are generic over RingClass (defined in core/paramatrix.jl).

_adj(c) = copy(c')

"""
    para(A) -> ParaMatrix
    paraconj(A)

The para-adjoint (para-conjugate) `Ã(z) = A(1/z̄)†` of a Laurent ParaMatrix: on the
unit circle it equals the conjugate-transpose, `para(A)(θ) = A(θ)'`. The
coefficient rule is `Ãₘ = (A₋ₘ)†` (the power window negates). Also reachable as `A'`.
"""
function para(A::ParaMatrix{T,S,<:Laurent}) where {T,S}
    c = A.class
    nc = Laurent(-c.hi, -c.lo)
    return ParaMatrix([_adj(coeff(A, -m)) for m in powers(nc)], nc)
end

"""
    paraconj(A)

Alias for the para-adjoint [`para`](@ref).
"""
paraconj(A) = para(A)

"""
    parahermitianpart(A) -> ParaMatrix

The para-Hermitian part `(A + para(A))/2` (so `isparahermitian` holds for the
result). Requires a symmetric window `Laurent(-L, L)` so that `A` and `para(A)`
share a class; this is the natural domain of para-Hermitian objects.
"""
function parahermitianpart(A::ParaMatrix{T,S,<:Laurent}) where {T,S}
    c = A.class
    c.lo == -c.hi || throw(
        ArgumentError(
            "parahermitianpart needs a symmetric window Laurent(-L,L); got $(c) " *
            "(the para-Hermitian part otherwise lives in the wider window Laurent(-$(max(c.hi,-c.lo)),$(max(c.hi,-c.lo))))",
        ),
    )
    return 0.5 * (A + para(A))
end

# on the circle the conjugate-transpose IS the para-adjoint
Base.adjoint(A::ParaMatrix{T,S,<:Laurent}) where {T,S} = para(A)

# predicates (Laurent / on-the-circle notions; surfaced under the LinearAlgebra names)
"""
    isparahermitian(A) -> Bool

Whether `para(A) == A` (Hermitian on the unit circle).
"""
function isparahermitian(A::ParaMatrix{T,S,<:Laurent}; tol=1e-9) where {T,S}
    Ã = para(A)
    Ã.class == A.class || return false
    return all(norm(A.coeffs[j] - Ã.coeffs[j]) ≤ tol for j in 1:nterms(A))
end

"""
    isparaunitary(A) -> Bool

Whether `para(A) * A` is the identity para-matrix — i.e. **left** para-unitary
(`Aᴴ(θ)A(θ) = I` on the circle). For square `A` this is two-sided unitarity; for
a tall `A` it is column-isometry only (and `inv` is then invalid).
"""
function isparaunitary(A::ParaMatrix{T,S,<:Laurent}; tol=1e-9) where {T,S}
    P = para(A) * A
    return norm(coeff(P, 0) - I) ≤ tol &&
           all(norm(coeff(P, k)) ≤ tol for k in powers(P.class) if k != 0)
end

"""
    ispositive(A; nsample, tol=-1e-9) -> Bool

Whether the minimum eigenvalue of `A(θ)` is `≥ tol` (default `-1e-9`, a
numerically tolerant PSD check) for all sampled `θ` on the circle.
"""
function ispositive(A::ParaMatrix{T,S,<:Laurent}; nsample=64, tol=-1e-9) where {T,S}
    return all(
        minimum(real, eigvals(Hermitian(Matrix(A(t))))) ≥ tol for
        t in range(0, 1; length=nsample + 1)[1:nsample]
    )
end

"""
    opnorm(A; nsample=256) -> Real

The H∞ / sup operator norm `max_θ ‖A(θ)‖₂` over the circle (the largest singular
value of `A(θ)` maximised on an `nsample` grid) — the gain of `A` as a
multiplication operator. Equals `1` for a para-unitary `A`.
"""
function LinearAlgebra.opnorm(A::ParaMatrix{T,S,<:Laurent}; nsample::Int=256) where {T,S}
    return maximum(opnorm(Matrix(A(t))) for t in range(0, 1; length=nsample + 1)[1:nsample])
end

function LinearAlgebra.ishermitian(A::ParaMatrix{T,S,<:Laurent}; kw...) where {T,S}
    return isparahermitian(A; kw...)
end
function LinearAlgebra.isposdef(A::ParaMatrix{T,S,<:Laurent}; kw...) where {T,S}
    return ispositive(A; kw...)
end

# det(A(z)) — a scalar Laurent polynomial — via evaluate on the circle + inverse DFT
function LinearAlgebra.det(A::ParaMatrix{T,S,<:Laurent}) where {T,S}
    d = size(A, 1)
    c = A.class
    lo, hi = d * c.lo, d * c.hi
    M = hi - lo + 1
    CT = complex(float(T))
    g = [cispi(-2 * lo * (j / M)) * det(Matrix{CT}(A(j / M))) for j in 0:(M - 1)]
    dk = [sum(g[j + 1] * cispi(-2 * j * m / M) for j in 0:(M - 1)) / M for m in 0:(M - 1)]
    return ParaMatrix([fill(dk[m + 1], 1, 1) for m in 0:(M - 1)], Laurent(lo, hi))
end

# inverse: EXACT and in-class for para-unitary (A⁻¹ = para(A)); otherwise the
# inverse is rational (adj(A)/det(A)) — return its best `Laurent(order)` fit
# (convergent when A⁻¹ is analytic on the circle, i.e. det A(θ) ≠ 0 there) and
# WARN if the fit does not converge. Use `pinv` for the sampled (pointwise) form.
function Base.inv(A::ParaMatrix{T,S,<:Laurent}; order::Int=8, tol=1e-9) where {T,S}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("inv needs a square ParaMatrix"))
    isparaunitary(A; tol=tol) && return para(A)
    x, info = para_solve(A, paraeye(size(A, 1), T, A.class); order=order)
    info.converged || @warn(
        "inv: A⁻¹ is rational (A is not para-unitary); the Laurent(order=$order) fit " *
            "has residual $(info.residual). Raise `order`, or use `pinv` for the sampled inverse.",
        maxlog = 3,
    )
    return x
end

# para-adjoint pullback (adjoint+mirror = para itself); kept with its primal
function ChainRulesCore.rrule(::typeof(para), A::ParaMatrix{T,S,<:Laurent}) where {T,S}
    Y = para(A)
    function para_back(Ȳ)
        c̄ = unthunk(Ȳ).coeffs
        Ā = para(ParaMatrix([copy(c) for c in c̄], Y.class)).coeffs
        return (NoTangent(), Tangent{typeof(A)}(; coeffs=Ā, class=NoTangent()))
    end
    return Y, para_back
end
