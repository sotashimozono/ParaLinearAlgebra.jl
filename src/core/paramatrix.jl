# core/paramatrix.jl — the central type and its class-agnostic algebra.
#
# A ParaMatrix is coefficients + a FunctionClass:  A(p) = Σ_k basis(class,p)_k · coeffs_k.
# Everything here is generic over the class through the `function_class.jl` interface;
# operations that need a specific structure (power window: *, kron, para, det, …) call
# class hooks (`powers`, `_prodclass`) that live in `classes/`.

"""
    ParaMatrix(coeffs::Vector{<:AbstractMatrix}, class::FunctionClass)

A matrix whose entries are functions of a scalar parameter `p` in `class`:
`A(p) = Σ_k basis(class, p)_k · coeffs_k`. All coefficient blocks share one size.
`A` is **callable**: `A(p)` materialises the dense matrix (so the whole
`LinearAlgebra` API works pointwise via `A(p)`).
"""
struct ParaMatrix{T,S<:AbstractMatrix{T},C<:FunctionClass}
    coeffs::Vector{S}
    class::C
    function ParaMatrix(
        coeffs::Vector{S}, class::C
    ) where {T,S<:AbstractMatrix{T},C<:FunctionClass}
        length(coeffs) == nbasis(class) ||
            error("coeff count $(length(coeffs)) ≠ basis size $(nbasis(class)) for $class")
        return new{T,S,C}(coeffs, class)
    end
end

Base.eltype(::ParaMatrix{T}) where {T} = T
Base.size(A::ParaMatrix) = size(A.coeffs[1])
Base.size(A::ParaMatrix, d::Int) = size(A.coeffs[1], d)

"""
    coefficients(A::ParaMatrix) -> Vector{<:AbstractMatrix}

The coefficient blocks — the AD parameter of the object.
"""
coefficients(A::ParaMatrix) = A.coeffs

"""
    function_class(A::ParaMatrix) -> FunctionClass
"""
function_class(A::ParaMatrix) = A.class

"""
    nterms(A::ParaMatrix) -> Int

Number of coefficient blocks (= `nbasis(function_class(A))`).
"""
nterms(A::ParaMatrix) = length(A.coeffs)

"""
    coeff(A::ParaMatrix, k::Int) -> AbstractMatrix

The coefficient block at integer power `k` (ring/power-window classes only).
"""
coeff(A::ParaMatrix, k::Int) = A.coeffs[k - first(powers(A.class)) + 1]

# ---- evaluation (the bridge to dense LinearAlgebra) ------------------------

"""
    evaluate(A::ParaMatrix, p) -> AbstractMatrix
    (A::ParaMatrix)(p)         -> AbstractMatrix

Materialise `A(p) = Σ_k basis(class, p)_k · coeffs_k`. `A(p)` is the callable form.
"""
evaluate(A::ParaMatrix, p) = sum(w * c for (w, c) in zip(basis(A.class, p), A.coeffs))
(A::ParaMatrix)(p) = evaluate(A, p)

"""
    evaluate_deriv(A::ParaMatrix, p) -> AbstractMatrix

The parameter derivative `∂_p A(p) = Σ_k basis_deriv(class, p)_k · coeffs_k`
(requires the class to define [`basis_deriv`](@ref)).
"""
evaluate_deriv(A::ParaMatrix, p) =
    sum(w * c for (w, c) in zip(basis_deriv(A.class, p), A.coeffs))

# ---- class-agnostic ring + structural operations --------------------------

_sameclass(A, B) =
    A.class == B.class || error("class mismatch: $(A.class) vs $(B.class)")

Base.:+(A::ParaMatrix, B::ParaMatrix) = (_sameclass(A, B); ParaMatrix(A.coeffs .+ B.coeffs, A.class))
Base.:-(A::ParaMatrix, B::ParaMatrix) = (_sameclass(A, B); ParaMatrix(A.coeffs .- B.coeffs, A.class))
Base.:-(A::ParaMatrix) = ParaMatrix([-c for c in A.coeffs], A.class)
Base.:*(α::Number, A::ParaMatrix) = ParaMatrix([α * c for c in A.coeffs], A.class)
Base.:*(A::ParaMatrix, α::Number) = α * A

Base.:(==)(A::ParaMatrix, B::ParaMatrix) = A.class == B.class && A.coeffs == B.coeffs
Base.isapprox(A::ParaMatrix, B::ParaMatrix; kw...) =
    A.class == B.class && all(isapprox(a, b; kw...) for (a, b) in zip(A.coeffs, B.coeffs))
Base.zero(A::ParaMatrix) = ParaMatrix([zero(c) for c in A.coeffs], A.class)
Base.copy(A::ParaMatrix) = ParaMatrix([copy(c) for c in A.coeffs], A.class)
Base.transpose(A::ParaMatrix) = ParaMatrix([copy(transpose(c)) for c in A.coeffs], A.class)

# matrix product / kron = coefficient convolution (windows add via `_prodclass`).
# Generic over the class — `powers`/`_prodclass` are the per-class hooks.
function _convolve(op, A::ParaMatrix, B::ParaMatrix)
    cls = _prodclass(A.class, B.class)
    pA, pB, pC = powers(A.class), powers(B.class), powers(cls)
    out = map(pC) do kc
        sum(
            op(A.coeffs[ia], B.coeffs[ib]) for ia in eachindex(pA) for
            ib in eachindex(pB) if pA[ia] + pB[ib] == kc
        )
    end
    return ParaMatrix(out, cls)
end
Base.:*(A::ParaMatrix, B::ParaMatrix) = _convolve(*, A, B)
Base.kron(A::ParaMatrix, B::ParaMatrix) = _convolve(kron, A, B)
const ⊗ = kron

function Base.:^(A::ParaMatrix, n::Integer)
    n ≥ 0 || error("negative powers are rational (out of the polynomial/Laurent class)")
    n == 0 && return one(A)
    B = A
    for _ in 2:n
        B = B * A
    end
    return B
end

# indexing: entry → 1×1 ParaMatrix ; block → sub ParaMatrix (same class)
Base.getindex(A::ParaMatrix, i::Int, j::Int) =
    ParaMatrix([fill(c[i, j], 1, 1) for c in A.coeffs], A.class)
Base.getindex(A::ParaMatrix, I::AbstractVector, J::AbstractVector) =
    ParaMatrix([c[I, J] for c in A.coeffs], A.class)

function Base.hcat(As::ParaMatrix...)
    cls = As[1].class
    all(A.class == cls for A in As) || error("hcat needs a common class")
    return ParaMatrix([reduce(hcat, (A.coeffs[k] for A in As)) for k in 1:nbasis(cls)], cls)
end
function Base.vcat(As::ParaMatrix...)
    cls = As[1].class
    all(A.class == cls for A in As) || error("vcat needs a common class")
    return ParaMatrix([reduce(vcat, (A.coeffs[k] for A in As)) for k in 1:nbasis(cls)], cls)
end

# class-agnostic reductions
LinearAlgebra.tr(A::ParaMatrix) = ParaMatrix([fill(tr(c), 1, 1) for c in A.coeffs], A.class)
LinearAlgebra.norm(A::ParaMatrix, p::Real=2) =
    p == 2 ? sqrt(sum(norm(c)^2 for c in A.coeffs)) :
    error("only the L² (Parseval) norm is defined for a ParaMatrix")
