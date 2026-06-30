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
        isempty(coeffs) && error("ParaMatrix needs at least one coefficient block")
        sz = size(first(coeffs))
        all(c -> size(c) == sz, coeffs) ||
            error("coefficient blocks must share one size; got $(unique(size.(coeffs)))")
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

The coefficient block at integer power `k` (ring classes only).
"""
function coeff(A::ParaMatrix, k::Int)
    function_class(A) isa RingClass ||
        error("coeff requires a ring class; got $(typeof(function_class(A)))")
    return A.coeffs[k - first(powers(A.class)) + 1]
end

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
    evaluate_deriv(A::ParaMatrix, ps, dim::Integer) -> AbstractMatrix

The parameter derivative `∂_p A(p) = Σ_k basis_deriv(class, p)_k · coeffs_k`
(requires the class to define [`basis_deriv`](@ref)). For a multi-parameter
([`ProductClass`](@ref)) object, the partial derivative along axis `dim`.
"""
function evaluate_deriv(A::ParaMatrix, p)
    return sum(w * c for (w, c) in zip(basis_deriv(A.class, p), A.coeffs))
end
function evaluate_deriv(A::ParaMatrix, ps, dim::Integer)
    return sum(w * c for (w, c) in zip(basis_deriv(A.class, ps, dim), A.coeffs))
end

# ---- class-agnostic ring + structural operations --------------------------

_sameclass(A, B) = A.class == B.class || error("class mismatch: $(A.class) vs $(B.class)")

function Base.:+(A::ParaMatrix, B::ParaMatrix)
    return (_sameclass(A, B); ParaMatrix(A.coeffs .+ B.coeffs, A.class))
end
function Base.:-(A::ParaMatrix, B::ParaMatrix)
    return (_sameclass(A, B); ParaMatrix(A.coeffs .- B.coeffs, A.class))
end
Base.:-(A::ParaMatrix) = ParaMatrix([-c for c in A.coeffs], A.class)
Base.:*(α::Number, A::ParaMatrix) = ParaMatrix([α * c for c in A.coeffs], A.class)
Base.:*(A::ParaMatrix, α::Number) = α * A

Base.:(==)(A::ParaMatrix, B::ParaMatrix) = A.class == B.class && A.coeffs == B.coeffs
function Base.isapprox(A::ParaMatrix, B::ParaMatrix; kw...)
    return A.class == B.class &&
           all(isapprox(a, b; kw...) for (a, b) in zip(A.coeffs, B.coeffs))
end
Base.zero(A::ParaMatrix) = ParaMatrix([zero(c) for c in A.coeffs], A.class)
Base.copy(A::ParaMatrix) = ParaMatrix([copy(c) for c in A.coeffs], A.class)
Base.transpose(A::ParaMatrix) = ParaMatrix([copy(transpose(c)) for c in A.coeffs], A.class)

# matrix product / kron = coefficient convolution (windows add via `_prodclass`).
# Generic over the class — `powers`/`_prodclass` are the per-class hooks.
function _convolve(op, A::ParaMatrix, B::ParaMatrix)
    cls = _prodclass(A.class, B.class)
    pA, pB, pC = powers(A.class), powers(B.class), powers(cls)
    out = map(pC) do kc
        return sum(
            op(A.coeffs[ia], B.coeffs[ib]) for ia in eachindex(pA) for
            ib in eachindex(pB) if pA[ia] + pB[ib] == kc
        )
    end
    return ParaMatrix(out, cls)
end
# ring product / kron / power require a RingClass; an ansatz class (e.g. Fourier)
# gets a clear ErrorException instead of a cryptic MethodError inside `_convolve`.
# A single method + runtime guard avoids the dispatch ambiguity a `<:RingClass`
# overload would create against the unconstrained signature.
function _notring(A)
    return error(
        "ring operation needs a RingClass (Laurent/Polynomial/ProductClass); got " *
        "$(typeof(function_class(A))) — an ansatz class. Evaluate first (`A(p)`) and use LinearAlgebra.",
    )
end
_assert_ring(A) = function_class(A) isa RingClass || _notring(A)

function Base.:*(A::ParaMatrix, B::ParaMatrix)
    _assert_ring(A)
    _assert_ring(B)
    return _convolve(*, A, B)
end
function Base.kron(A::ParaMatrix, B::ParaMatrix)
    _assert_ring(A)
    _assert_ring(B)
    return _convolve(kron, A, B)
end
"""
    ⊗(A, B)

Infix Kronecker product `A ⊗ B = kron(A, B)` (coefficient-convolution kron of two
ParaMatrices).
"""
⊗(A, B) = kron(A, B)

function Base.:^(A::ParaMatrix, n::Integer)
    _assert_ring(A)
    n ≥ 0 || error("negative powers are rational (out of the polynomial/Laurent class)")
    n == 0 && return one(A)
    B = A
    for _ in 2:n
        B = B * A
    end
    return B
end

# identity element, generic over ring classes (the zero power carries `I`).
"""
    paraeye(d, T, class::RingClass) -> ParaMatrix

The `d×d` identity ParaMatrix (`I` at the zero power, zeros elsewhere), element type `T`.
"""
function paraeye(d::Int, ::Type{T}, class::RingClass) where {T}
    return ParaMatrix(
        [k == zero(k) ? Matrix{T}(I, d, d) : zeros(T, d, d) for k in powers(class)], class
    )
end
function Base.one(A::ParaMatrix)
    _assert_ring(A)
    size(A, 1) == size(A, 2) || error("one(A) needs a square ParaMatrix")
    pw = powers(function_class(A))
    any(k -> k == zero(k), pw) || error("one(A) needs the zero power in the window")
    return paraeye(size(A, 1), eltype(A), function_class(A))
end

# indexing: entry → 1×1 ParaMatrix ; block → sub ParaMatrix (same class)
function Base.getindex(A::ParaMatrix, i::Int, j::Int)
    return ParaMatrix([fill(c[i, j], 1, 1) for c in A.coeffs], A.class)
end
function Base.getindex(A::ParaMatrix, I::AbstractVector, J::AbstractVector)
    return ParaMatrix([c[I, J] for c in A.coeffs], A.class)
end

"""
    diag(A::ParaMatrix) -> Vector{<:ParaMatrix}

The diagonal as the vector of scalar (1×1) parameterized entries `A[i,i]`. For a
square `A`, `sum(diag(A)) == tr(A)`, and `diagm(diag(A))` rebuilds the diagonal part.
"""
LinearAlgebra.diag(A::ParaMatrix) = [A[i, i] for i in 1:minimum(size(A))]

"""
    diagm(ds::AbstractVector{<:ParaMatrix}) -> ParaMatrix

Build a diagonal ParaMatrix from scalar (1×1) parameterized entries of a common
class: `diagm(ds)(θ) = Diagonal([d(θ) for d in ds])`.
"""
function LinearAlgebra.diagm(ds::AbstractVector{<:ParaMatrix})
    isempty(ds) && error("diagm needs at least one entry")
    cls = function_class(first(ds))
    all(function_class(d) == cls for d in ds) || error("diagm entries need a common class")
    all(size(d) == (1, 1) for d in ds) || error("diagm entries must be 1×1 (scalar)")
    n = length(ds)
    coeffs = [Matrix(Diagonal([ds[i].coeffs[k][1, 1] for i in 1:n])) for k in 1:nbasis(cls)]
    return ParaMatrix(coeffs, cls)
end

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

"""
    dot(A::ParaMatrix, B::ParaMatrix) -> Number

The `L²` (Frobenius-integrated) inner product of two same-class parameterized
matrices, `⟨A,B⟩ = ∫₀¹ ⟨A(θ),B(θ)⟩_F dθ = Σ_{kl} M_{kl} ⟨Aₖ,Bₗ⟩_F`, with the
basis Gram `M = basis_gram(class)` (uniform measure on the torus). This is the
substrate for parameter-geometry: Berry connection `⟨ψ,∂_θ ψ⟩` and the
Fubini–Study / quantum metric are built from `dot` of states and their
`evaluate_deriv`s by a downstream layer.
"""
function LinearAlgebra.dot(A::ParaMatrix, B::ParaMatrix)
    A.class == B.class || error("dot needs a common class: $(A.class) vs $(B.class)")
    M = basis_gram(A.class)
    n = nterms(A)
    return sum(M[k, l] * dot(A.coeffs[k], B.coeffs[l]) for k in 1:n, l in 1:n)
end

"""
    norm(A::ParaMatrix, p=2) -> Real

The `L²` function norm `‖A‖ = sqrt(∫₀¹ ‖A(θ)‖_F² dθ) = sqrt(real⟨A,A⟩)` (see
[`dot`](@ref)). For an orthonormal basis ([`Laurent`](@ref)) this is Parseval
`sqrt(Σ‖Aₖ‖²)`; for `Fourier`/`Polynomial` the Gram supplies the weight factors.
"""
function LinearAlgebra.norm(A::ParaMatrix, p::Real=2)
    p == 2 || error("only the L² norm (p=2) is defined for a ParaMatrix")
    return sqrt(max(real(dot(A, A)), 0.0))
end

"""
    integral(A::ParaMatrix) -> AbstractMatrix

The parameter integral `∫₀¹ A(θ) dθ = Σ_k basis_integral(class)_k · coeffsₖ`
(uniform measure). For [`Laurent`](@ref) this is the zero-mode `coeff(A, 0)`; for
[`Fourier`](@ref) the constant term. The mean of `A(θ)` over the torus.
"""
function integral(A::ParaMatrix)
    return sum(w * c for (w, c) in zip(basis_integral(A.class), A.coeffs))
end
