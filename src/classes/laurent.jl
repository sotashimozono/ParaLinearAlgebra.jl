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
struct Laurent <: FunctionClass
    lo::Int
    hi::Int
    function Laurent(lo::Int, hi::Int)
        lo ≤ hi || throw(ArgumentError("Laurent window needs lo ≤ hi; got $lo:$hi"))
        return new(lo, hi)
    end
end
Laurent(L::Int) = Laurent(-L, L)
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

# identity / one
"""
    paraeye(d, T, class::Laurent) -> ParaMatrix

The `d×d` identity ParaMatrix (`I` at power 0) of element type `T`.
"""
function paraeye(d::Int, ::Type{T}, class::Laurent) where {T}
    return ParaMatrix(
        [k == 0 ? Matrix{T}(I, d, d) : zeros(T, d, d) for k in powers(class)], class
    )
end
function Base.one(A::ParaMatrix{T,S,<:Laurent}) where {T,S}
    size(A, 1) == size(A, 2) || error("one(A) needs a square ParaMatrix")
    0 in powers(A.class) || error("one(A) needs 0 in the power window")
    return paraeye(size(A, 1), T, A.class)
end

# para-adjoint  Ã(z) = A(1/z̄)†  ⇒  Ã_m = (A_{-m})†, window negates
_adj(c) = copy(c')
function para(A::ParaMatrix{T,S,<:Laurent}) where {T,S}
    c = A.class
    nc = Laurent(-c.hi, -c.lo)
    return ParaMatrix([_adj(coeff(A, -m)) for m in powers(nc)], nc)
end
const paraconj = para

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

Whether `para(A) * A` is the identity para-matrix (unitary on the circle).
"""
function isparaunitary(A::ParaMatrix{T,S,<:Laurent}; tol=1e-9) where {T,S}
    P = para(A) * A
    return norm(coeff(P, 0) - I) ≤ tol &&
           all(norm(coeff(P, k)) ≤ tol for k in powers(P.class) if k != 0)
end

"""
    ispositive(A; nsample) -> Bool

Whether `A(θ) ⪰ 0` for all sampled `θ` on the circle.
"""
function ispositive(A::ParaMatrix{T,S,<:Laurent}; nsample=64, tol=-1e-9) where {T,S}
    return all(
        minimum(real, eigvals(Hermitian(Matrix(A(t))))) ≥ tol for
        t in range(0, 1; length=nsample + 1)[1:nsample]
    )
end

LinearAlgebra.ishermitian(A::ParaMatrix{T,S,<:Laurent}; kw...) where {T,S} =
    isparahermitian(A; kw...)
LinearAlgebra.isposdef(A::ParaMatrix{T,S,<:Laurent}; kw...) where {T,S} =
    ispositive(A; kw...)

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

# inverse — in-class only for para-unitary (Ã = A⁻¹); the general inverse is rational
function Base.inv(A::ParaMatrix{T,S,<:Laurent}; tol=1e-9) where {T,S}
    isparaunitary(A; tol=tol) ||
        error("only the para-unitary inverse is in-class (= para); a general inverse is rational")
    return para(A)
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
