# ParaLinearAlgebraPolynomialsExt — the Polynomials.jl bridge (loaded automatically
# when both ParaLinearAlgebra and Polynomials are present).
#
# It adds `PolynomialsBasis`: a `FunctionClass` whose basis is the first N+1 basis
# polynomials of a Polynomials.jl family (`ChebyshevT`, `Polynomials.Polynomial`,
# or a `SpecialPolynomials.jl` family such as `Legendre`/`Hermite`/`Laguerre`),
# with EXACT derivatives and L² Gram/integral supplied by Polynomials.jl. Because
# an orthogonal family does not multiply by index-convolution, this is an ANSATZ
# class; `monomialize` re-expresses it (exactly) in the monomial `Polynomial` ring
# so the products become available. Two conversions bridge scalar ParaMatrices and
# `Polynomials.Polynomial`.
module ParaLinearAlgebraPolynomialsExt

using ParaLinearAlgebra
using ParaLinearAlgebra: AbstractParaMatrix, FunctionClass, coefficients, function_class
using Polynomials: Polynomials

"""
    PolynomialsBasis <: FunctionClass

The class returned by [`polynomial_basis`](@ref): it stores the actual
Polynomials.jl basis polynomials `b₀,…,b_N` of a family and its L² domain `[a,b]`.
An ansatz class (no `*`/`kron`); use [`monomialize`](@ref) for the ring.
"""
struct PolynomialsBasis{TP<:Polynomials.AbstractPolynomial} <: FunctionClass
    polys::Vector{TP}
    a::Float64
    b::Float64
    label::String
end

# the k-th basis element of family P is the family polynomial with unit coefficient
# in slot k — `P([0,…,0,1])`. Robust across Polynomials.jl versions.
function _basis_poly(::Type{P}, k::Integer) where {P<:Polynomials.AbstractPolynomial}
    return P([j == k ? 1 : 0 for j in 0:k])
end

# monomial (standard-basis) form of every basis polynomial — the change-of-basis data
_monos(c::PolynomialsBasis) = [convert(Polynomials.Polynomial, p) for p in c.polys]

# definite integral ∫ₐᵇ q via the antiderivative (integrate/1 is universal across versions)
function _defint(q::Polynomials.AbstractPolynomial, a::Real, b::Real)
    Q = Polynomials.integrate(q)
    return Q(b) - Q(a)
end

_conjp(p::Polynomials.Polynomial) = Polynomials.Polynomial(conj.(Polynomials.coeffs(p)))

ParaLinearAlgebra.nbasis(c::PolynomialsBasis) = length(c.polys)
ParaLinearAlgebra.basis(c::PolynomialsBasis, x) = [p(x) for p in c.polys]
function ParaLinearAlgebra.basis_deriv(c::PolynomialsBasis, x)
    return [Polynomials.derivative(p)(x) for p in c.polys]
end

# L² Gram Mₖₗ = ∫ conj(bₖ) bₗ and integral ∫ bₖ over [a,b] — EXACT (both are polynomials)
function ParaLinearAlgebra.basis_gram(c::PolynomialsBasis)
    mono = _monos(c)
    n = length(mono)
    return [_defint(_conjp(mono[k]) * mono[l], c.a, c.b) for k in 1:n, l in 1:n]
end
function ParaLinearAlgebra.basis_integral(c::PolynomialsBasis)
    return [_defint(p, c.a, c.b) for p in _monos(c)]
end

# two `PolynomialsBasis` are equal when they span the same family window on the same
# domain (label is cosmetic) — so operands built independently still combine.
function Base.:(==)(x::PolynomialsBasis, y::PolynomialsBasis)
    return x.a == y.a && x.b == y.b && x.polys == y.polys
end

function Base.show(io::IO, c::PolynomialsBasis)
    return print(io, "PolynomialsBasis(", c.label, " on [", c.a, ",", c.b, "])")
end

function _default_interval(::Type{P}, interval) where {P<:Polynomials.AbstractPolynomial}
    interval !== nothing && return (float(first(interval)), float(last(interval)))
    P === Polynomials.Polynomial && return (0.0, 1.0)   # match the core monomial `Polynomial`
    return (-1.0, 1.0)                                   # standard orthogonal-poly domain
end

function ParaLinearAlgebra.polynomial_basis(
    ::Type{P}, N::Integer; interval=nothing, label=nothing
) where {P<:Polynomials.AbstractPolynomial}
    N ≥ 0 || throw(ArgumentError("polynomial_basis degree N must be ≥ 0; got $N"))
    polys = [_basis_poly(P, k) for k in 0:N]
    a, b = _default_interval(P, interval)
    a < b || throw(ArgumentError("interval must be (a,b) with a < b; got ($a, $b)"))
    lab = label === nothing ? "$(nameof(P))(0:$N)" : String(label)
    return PolynomialsBasis(polys, a, b, lab)
end

# exact change of basis into the monomial `Polynomial` ring:
#   A(x) = Σ_k bₖ(x) Cₖ,   bₖ(x) = Σ_j mₖⱼ xʲ   ⇒   A(x) = Σ_j (Σ_k mₖⱼ Cₖ) xʲ.
function ParaLinearAlgebra.monomialize(
    A::AbstractParaMatrix{T,S,<:PolynomialsBasis}
) where {T,S}
    c = function_class(A)
    mono = _monos(c)
    N = maximum(Polynomials.degree, mono)
    C = coefficients(A)
    d1, d2 = size(A)
    Tp = promote_type(eltype(first(C)), eltype(Polynomials.coeffs(first(mono))))
    D = [zeros(Tp, d1, d2) for _ in 0:N]
    for k in eachindex(mono)
        ck = Polynomials.coeffs(mono[k])          # ck[j] = coefficient of x^{j-1}
        for j in eachindex(ck)
            @. D[j] += ck[j] * C[k]               # D index j ↔ power j-1
        end
    end
    return ParaMatrix(D, Polynomial(N))
end

# ---- conversions: scalar monomial ParaMatrix ⇄ Polynomials.Polynomial ----

"""
    Polynomials.Polynomial(A::AbstractParaMatrix) -> Polynomials.Polynomial

The scalar (1×1) monomial [`Polynomial`](@ref)-class ParaMatrix `A` as a
`Polynomials.Polynomial` (coefficients = `A`'s coefficient blocks). Errors if `A`
is not scalar.
"""
function Polynomials.Polynomial(
    A::AbstractParaMatrix{T,S,<:ParaLinearAlgebra.Polynomial}
) where {T,S}
    size(A) == (1, 1) || throw(
        ArgumentError(
            "Polynomials.Polynomial(A) needs a scalar (1×1) ParaMatrix; got size $(size(A))",
        ),
    )
    return Polynomials.Polynomial([only(c) for c in coefficients(A)])
end

"""
    ParaMatrix(p::Polynomials.AbstractPolynomial) -> ParaMatrix

A scalar (1×1) ParaMatrix in the monomial [`Polynomial`](@ref) ring class equal to
`p` (any Polynomials.jl family is first linearised to the monomial basis, so a
`ChebyshevT` becomes its monomial expansion).
"""
function ParaLinearAlgebra.ParaMatrix(p::Polynomials.AbstractPolynomial)
    mono = convert(Polynomials.Polynomial, p)
    cs = Polynomials.coeffs(mono)
    isempty(cs) && (cs = [zero(eltype(mono))])
    return ParaMatrix([fill(c, 1, 1) for c in cs], Polynomial(length(cs) - 1))
end

end # module ParaLinearAlgebraPolynomialsExt
