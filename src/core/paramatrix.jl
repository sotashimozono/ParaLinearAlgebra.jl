# core/paramatrix.jl — the central type and its class-agnostic algebra.
#
# A ParaMatrix is coefficients + a FunctionClass:  A(p) = Σ_k basis(class,p)_k · coeffs_k.
# Everything here is generic over the class through the `function_class.jl` interface;
# operations that need a specific structure (power window: *, kron, para, det, …) call
# class hooks (`powers`, `_prodclass`) that live in `classes/`.

"""
    AbstractParaMatrix{T,S,C}

Supertype of every parameterized matrix: element type `T`, coefficient-block
storage `S<:AbstractMatrix{T}`, function class `C<:FunctionClass`. The whole
algebra (evaluation, `+`/`-`/`*`/`kron`, `para`, `det`, `inv`, the factorizations,
…) is written against this supertype, so a **user-defined subtype gets it all for
free** by implementing the small interface below.

# Interface (what a subtype must / may define)
- `coefficients(A)` **(required)** → the `Vector` of coefficient blocks.
- `function_class(A)` **(required)** → the `FunctionClass`.
- `_rebuild(A, coeffs, class)` *(optional)* → build a like-kind object from new
  `coeffs`/`class`; **override to make the operations return your own type**
  (default returns a plain [`ParaMatrix`](@ref)).
- `evaluate(A, p)` / `(A)(p)` *(optional)* → default is `Σ basis(class,p)_k·coeffs_k`;
  override only if your storage evaluates specially.

Given `coefficients` + `function_class`, everything else (`size`, `eltype`,
`nterms`, `coeff`, `evaluate`, the ring/structural ops, and — for a `Laurent`
class — `para`/`det`/`inv`/`eigen`/`svd`/`para_qr`/… ) works automatically.
"""
abstract type AbstractParaMatrix{T,S,C} end

"""
    ParaMatrix(coeffs::Vector{<:AbstractMatrix}, class::FunctionClass)

The canonical [`AbstractParaMatrix`](@ref): a matrix whose entries are functions of
a parameter `p` in `class`, `A(p) = Σ_k basis(class, p)_k · coeffs_k`. All
coefficient blocks share one size. `A` is **callable**: `A(p)` materialises the
dense matrix (so the whole `LinearAlgebra` API works pointwise via `A(p)`).
"""
struct ParaMatrix{T,S<:AbstractMatrix{T},C<:FunctionClass} <: AbstractParaMatrix{T,S,C}
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

# ---- interface -------------------------------------------------------------

"""
    coefficients(A::AbstractParaMatrix) -> Vector{<:AbstractMatrix}

The coefficient blocks — the AD parameter of the object. **Interface method.**
"""
coefficients(A::ParaMatrix) = A.coeffs

"""
    function_class(A::AbstractParaMatrix) -> FunctionClass

The function class. **Interface method.**
"""
function_class(A::ParaMatrix) = A.class

# reconstruction hook: how an operation builds a like-kind result. Override for a
# custom subtype to keep your own type through +, *, para, … (default: ParaMatrix).
_rebuild(::AbstractParaMatrix, coeffs, class) = ParaMatrix(coeffs, class)

Base.eltype(::AbstractParaMatrix{T}) where {T} = T
Base.size(A::AbstractParaMatrix) = size(coefficients(A)[1])
Base.size(A::AbstractParaMatrix, d::Int) = size(coefficients(A)[1], d)

"""
    nterms(A::AbstractParaMatrix) -> Int

Number of coefficient blocks (= `nbasis(function_class(A))`).
"""
nterms(A::AbstractParaMatrix) = length(coefficients(A))

"""
    coeff(A::AbstractParaMatrix, k::Int) -> AbstractMatrix

The coefficient block at integer power `k` (ring classes only).
"""
function coeff(A::AbstractParaMatrix, k::Int)
    function_class(A) isa RingClass ||
        error("coeff requires a ring class; got $(typeof(function_class(A)))")
    return coefficients(A)[k - first(powers(function_class(A))) + 1]
end

# ---- evaluation (the bridge to dense LinearAlgebra) ------------------------

"""
    evaluate(A::AbstractParaMatrix, p) -> AbstractMatrix
    (A::AbstractParaMatrix)(p)         -> AbstractMatrix

Materialise `A(p) = Σ_k basis(class, p)_k · coeffs_k`. `A(p)` is the callable form.
"""
function evaluate(A::AbstractParaMatrix, p)
    return sum(w * c for (w, c) in zip(basis(function_class(A), p), coefficients(A)))
end
(A::AbstractParaMatrix)(p) = evaluate(A, p)

"""
    evaluate_deriv(A::AbstractParaMatrix, p) -> AbstractMatrix
    evaluate_deriv(A::AbstractParaMatrix, ps, dim::Integer) -> AbstractMatrix

The parameter derivative `∂_p A(p) = Σ_k basis_deriv(class, p)_k · coeffs_k`
(requires the class to define [`basis_deriv`](@ref)). For a multi-parameter
([`ProductClass`](@ref)) object, the partial derivative along axis `dim`.
"""
function evaluate_deriv(A::AbstractParaMatrix, p)
    return sum(w * c for (w, c) in zip(basis_deriv(function_class(A), p), coefficients(A)))
end
function evaluate_deriv(A::AbstractParaMatrix, ps, dim::Integer)
    return sum(
        w * c for (w, c) in zip(basis_deriv(function_class(A), ps, dim), coefficients(A))
    )
end

# ---- class-agnostic ring + structural operations --------------------------

function _sameclass(A, B)
    return function_class(A) == function_class(B) ||
           error("class mismatch: $(function_class(A)) vs $(function_class(B))")
end

function Base.:+(A::AbstractParaMatrix, B::AbstractParaMatrix)
    _sameclass(A, B)
    return _rebuild(A, coefficients(A) .+ coefficients(B), function_class(A))
end
function Base.:-(A::AbstractParaMatrix, B::AbstractParaMatrix)
    _sameclass(A, B)
    return _rebuild(A, coefficients(A) .- coefficients(B), function_class(A))
end
function Base.:-(A::AbstractParaMatrix)
    return _rebuild(A, [-c for c in coefficients(A)], function_class(A))
end
function Base.:*(α::Number, A::AbstractParaMatrix)
    return _rebuild(A, [α * c for c in coefficients(A)], function_class(A))
end
Base.:*(A::AbstractParaMatrix, α::Number) = α * A
function Base.:/(A::AbstractParaMatrix, α::Number)
    return _rebuild(A, [c / α for c in coefficients(A)], function_class(A))
end

function Base.:(==)(A::AbstractParaMatrix, B::AbstractParaMatrix)
    return function_class(A) == function_class(B) && coefficients(A) == coefficients(B)
end
function Base.isapprox(A::AbstractParaMatrix, B::AbstractParaMatrix; kw...)
    return function_class(A) == function_class(B) &&
           all(isapprox(a, b; kw...) for (a, b) in zip(coefficients(A), coefficients(B)))
end
function Base.zero(A::AbstractParaMatrix)
    return _rebuild(A, [zero(c) for c in coefficients(A)], function_class(A))
end
function Base.copy(A::AbstractParaMatrix)
    return _rebuild(A, [copy(c) for c in coefficients(A)], function_class(A))
end
function Base.transpose(A::AbstractParaMatrix)
    return _rebuild(A, [copy(transpose(c)) for c in coefficients(A)], function_class(A))
end

# matrix product / kron = coefficient convolution (windows add via `_prodclass`).
# Generic over the class — `powers`/`_prodclass` are the per-class hooks.
function _convolve(op, A::AbstractParaMatrix, B::AbstractParaMatrix)
    cls = _prodclass(function_class(A), function_class(B))
    cA, cB = coefficients(A), coefficients(B)
    pA, pB, pC = powers(function_class(A)), powers(function_class(B)), powers(cls)
    out = map(pC) do kc
        return sum(
            op(cA[ia], cB[ib]) for ia in eachindex(pA) for
            ib in eachindex(pB) if pA[ia] + pB[ib] == kc
        )
    end
    return _rebuild(A, out, cls)
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

function Base.:*(A::AbstractParaMatrix, B::AbstractParaMatrix)
    _assert_ring(A)
    _assert_ring(B)
    return _convolve(*, A, B)
end
function Base.kron(A::AbstractParaMatrix, B::AbstractParaMatrix)
    _assert_ring(A)
    _assert_ring(B)
    return _convolve(kron, A, B)
end
"""
    ⊗(A, B, Cs...)

Infix Kronecker product `A ⊗ B = kron(A, B)` (coefficient-convolution kron of two
ParaMatrices). Variadic: `A ⊗ B ⊗ C` folds left.
"""
⊗(A, B, Cs...) = foldl(kron, (A, B, Cs...))

"""
    directsum(A, B) -> AbstractParaMatrix
    A ⊕ B

Direct sum (block diagonal): `(A ⊕ B)(θ) = [A(θ) 0; 0 B(θ)]`. Purely structural —
defined for any (matching) class, no ring structure needed — and it composes the
spectra and determinants of the parts (`eig(A⊕B) = eig(A) ∪ eig(B)`,
`det(A⊕B) = det(A)·det(B)`). The companion of the Kronecker product `⊗`.
"""
function directsum(A::AbstractParaMatrix, B::AbstractParaMatrix)
    _sameclass(A, B)
    coeffs = [cat(a, b; dims=(1, 2)) for (a, b) in zip(coefficients(A), coefficients(B))]
    return _rebuild(A, coeffs, function_class(A))
end

"""
    ⊕(A, B, Cs...)

Infix direct sum `A ⊕ B = directsum(A, B)` (block-diagonal stack). Variadic:
`A ⊕ B ⊕ C` folds left.
"""
⊕(A, B, Cs...) = foldl(directsum, (A, B, Cs...))

function Base.:^(A::AbstractParaMatrix, n::Integer)
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
function Base.one(A::AbstractParaMatrix)
    _assert_ring(A)
    size(A, 1) == size(A, 2) || error("one(A) needs a square ParaMatrix")
    cls = function_class(A)
    pw = powers(cls)
    any(k -> k == zero(k), pw) || error("one(A) needs the zero power in the window")
    d = size(A, 1)
    coeffs = [
        k == zero(k) ? Matrix{eltype(A)}(I, d, d) : zeros(eltype(A), d, d) for k in pw
    ]
    return _rebuild(A, coeffs, cls)
end

# indexing: entry → 1×1 ParaMatrix ; block → sub ParaMatrix (same class)
function Base.getindex(A::AbstractParaMatrix, i::Int, j::Int)
    return _rebuild(A, [fill(c[i, j], 1, 1) for c in coefficients(A)], function_class(A))
end
function Base.getindex(A::AbstractParaMatrix, I::AbstractVector, J::AbstractVector)
    return _rebuild(A, [c[I, J] for c in coefficients(A)], function_class(A))
end

"""
    diag(A::AbstractParaMatrix) -> Vector{<:AbstractParaMatrix}

The diagonal as the vector of scalar (1×1) parameterized entries `A[i,i]`. For a
square `A`, `sum(diag(A)) == tr(A)`, and `diagm(diag(A))` rebuilds the diagonal part.
"""
LinearAlgebra.diag(A::AbstractParaMatrix) = [A[i, i] for i in 1:minimum(size(A))]

"""
    diagm(ds::AbstractVector{<:AbstractParaMatrix}) -> AbstractParaMatrix

Build a diagonal ParaMatrix from scalar (1×1) parameterized entries of a common
class: `diagm(ds)(θ) = Diagonal([d(θ) for d in ds])`.
"""
function LinearAlgebra.diagm(ds::AbstractVector{<:AbstractParaMatrix})
    isempty(ds) && error("diagm needs at least one entry")
    cls = function_class(first(ds))
    all(function_class(d) == cls for d in ds) || error("diagm entries need a common class")
    all(size(d) == (1, 1) for d in ds) || error("diagm entries must be 1×1 (scalar)")
    n = length(ds)
    coeffs = [
        Matrix(Diagonal([coefficients(ds[i])[k][1, 1] for i in 1:n])) for k in 1:nbasis(cls)
    ]
    return _rebuild(first(ds), coeffs, cls)
end

function Base.hcat(As::AbstractParaMatrix...)
    cls = function_class(As[1])
    all(function_class(A) == cls for A in As) || error("hcat needs a common class")
    coeffs = [reduce(hcat, (coefficients(A)[k] for A in As)) for k in 1:nbasis(cls)]
    return _rebuild(As[1], coeffs, cls)
end
function Base.vcat(As::AbstractParaMatrix...)
    cls = function_class(As[1])
    all(function_class(A) == cls for A in As) || error("vcat needs a common class")
    coeffs = [reduce(vcat, (coefficients(A)[k] for A in As)) for k in 1:nbasis(cls)]
    return _rebuild(As[1], coeffs, cls)
end
# block-matrix literal `[A B; C D]` — per-coefficient hvcat with the same row spec
function Base.hvcat(rows::Tuple{Vararg{Int}}, As::AbstractParaMatrix...)
    cls = function_class(As[1])
    all(function_class(A) == cls for A in As) || error("hvcat needs a common class")
    coeffs = [hvcat(rows, (coefficients(A)[k] for A in As)...) for k in 1:nbasis(cls)]
    return _rebuild(As[1], coeffs, cls)
end

# class-agnostic reductions
function LinearAlgebra.tr(A::AbstractParaMatrix)
    return _rebuild(A, [fill(tr(c), 1, 1) for c in coefficients(A)], function_class(A))
end

"""
    dot(A::AbstractParaMatrix, B::AbstractParaMatrix) -> Number

The `L²` (Frobenius-integrated) inner product of two same-class parameterized
matrices, `⟨A,B⟩ = ∫₀¹ ⟨A(θ),B(θ)⟩_F dθ = Σ_{kl} M_{kl} ⟨Aₖ,Bₗ⟩_F`, with the
basis Gram `M = basis_gram(class)` (uniform measure on the torus). Together with
`evaluate_deriv` it is the substrate a downstream layer can use for differential
geometry of the parameterization (inner products of states and their
parameter-derivatives).
"""
function LinearAlgebra.dot(A::AbstractParaMatrix, B::AbstractParaMatrix)
    function_class(A) == function_class(B) ||
        error("dot needs a common class: $(function_class(A)) vs $(function_class(B))")
    M = basis_gram(function_class(A))
    cA, cB = coefficients(A), coefficients(B)
    n = nterms(A)
    return sum(M[k, l] * dot(cA[k], cB[l]) for k in 1:n, l in 1:n)
end

"""
    norm(A::AbstractParaMatrix, p=2) -> Real

The `L²` function norm `‖A‖ = sqrt(∫₀¹ ‖A(θ)‖_F² dθ) = sqrt(real⟨A,A⟩)` (see
`dot`). For an orthonormal basis ([`Laurent`](@ref)) this is Parseval
`sqrt(Σ‖Aₖ‖²)`; for `Fourier`/`Polynomial` the Gram supplies the weight factors.
"""
function LinearAlgebra.norm(A::AbstractParaMatrix, p::Real=2)
    p == 2 || error("only the L² norm (p=2) is defined for a ParaMatrix")
    return sqrt(max(real(dot(A, A)), 0.0))
end

"""
    integral(A::AbstractParaMatrix) -> AbstractMatrix

The parameter integral `∫₀¹ A(θ) dθ = Σ_k basis_integral(class)_k · coeffsₖ`
(uniform measure). For [`Laurent`](@ref) this is the zero-mode `coeff(A, 0)`; for
[`Fourier`](@ref) the constant term. The mean of `A(θ)` over the torus.
"""
function integral(A::AbstractParaMatrix)
    return sum(w * c for (w, c) in zip(basis_integral(function_class(A)), coefficients(A)))
end
