# core/function_class.jl — the abstract parameter axis.
#
# A `FunctionClass` is the ONLY thing the core knows about parameter dependence.
# Concrete classes (Fourier, Laurent, Polynomial, …) live in `classes/` and are
# plug-ins: the core never names them. A class C with parameter p materialises an
# object as  X(p) = Σ_k basis(C, p)_k · coeffs_k.
#
# Interface a concrete class must implement (in classes/<name>.jl):
#   REQUIRED  basis(c, p)        -> AbstractVector   weights w(p)
#   REQUIRED  nbasis(c)          -> Int              == length(basis(c, p))
#   OPTIONAL  basis_deriv(c, p)  -> AbstractVector   dw/dp  (enables ∂_p / geometry)
#   OPTIONAL  powers(c)          -> indices          integer power window
#                                  (ring classes only: enables *, kron, coeff, para …)

"""
    FunctionClass

Abstract supertype for the parameter-dependence of a parameterized object.
Concrete classes implement [`basis`](@ref) and [`nbasis`](@ref); optionally
[`basis_deriv`](@ref), [`basis_gram`](@ref), and [`powers`](@ref). The core is
generic over this interface — individual classes are defined separately in
`classes/`.

Two kinds of class:
- **ansatz** classes (e.g. [`Fourier`](@ref)) — support construction,
  `evaluate`, `∂_p`, and AD, but NOT the multiplication ring.
- **ring** classes ([`RingClass`](@ref): [`Laurent`](@ref)/[`Analytic`](@ref),
  [`Polynomial`](@ref), [`ProductClass`](@ref)) — additionally form a ring under
  `*`/`kron`/`^` and provide [`powers`](@ref)/`coeff`/`one`.
"""
abstract type FunctionClass end

"""
    RingClass <: FunctionClass

Classes whose ParaMatrices form a multiplication ring: products of factors are
again ParaMatrices of a (combined) class. Ring classes define [`powers`](@ref)
and a [`_prodclass`](@ref) rule, which enable `*`, `kron`, `^`, `coeff`, `one`,
and `paraeye`. Non-ring (ansatz) classes deliberately lack these.
"""
abstract type RingClass <: FunctionClass end

"""
    basis(c::FunctionClass, p) -> AbstractVector

The weights `w(p)` of class `c` at parameter `p`.
"""
function basis end

"""
    basis_deriv(c::FunctionClass, p) -> AbstractVector
    basis_deriv(c::FunctionClass, p, dim::Integer) -> AbstractVector

The parameter derivative `dw/dp` of the weights — the primitive behind
parameter-derivatives of any object built on `c` (sensitivities; differential
geometry of the parameterization). Optional; defined by differentiable classes. For a
multi-parameter class ([`ProductClass`](@ref)) the partial derivative along axis
`dim` is `basis_deriv(c, ps, dim)`; the scalar form is undefined there.
"""
function basis_deriv end

# single-parameter classes: only dim 1 exists
function basis_deriv(c::FunctionClass, p, dim::Integer)
    if dim == 1
        basis_deriv(c, p)
    else
        throw(ArgumentError("$(typeof(c)) has one parameter; dim must be 1, got $dim"))
    end
end

"""
    basis_gram(c::FunctionClass) -> AbstractMatrix

The `nbasis × nbasis` Gram matrix `M_{kl} = ∫ conj(wₖ(θ)) wₗ(θ) dθ` of the basis
over one period (uniform measure) — the metric that turns coefficient inner
products into the `L²` function inner product, `⟨A,B⟩_{L²} = Σ_{kl} M_{kl} ⟨Aₖ,Bₗ⟩_F`.
Used by `dot` and `norm`; `M = I` for an L²-orthonormal basis
([`Laurent`](@ref)).
"""
function basis_gram end

"""
    basis_integral(c::FunctionClass) -> AbstractVector

The integrals `∫₀¹ wₖ(θ) dθ` of the basis weights over one period (uniform
measure). Used by [`integral`](@ref): `∫₀¹ A(θ) dθ = Σ_k basis_integral(c)_k · coeffsₖ`.
"""
function basis_integral end

"""
    powers(c::FunctionClass)

The integer power window of a *ring* class (e.g. `k = lo:hi` for Laurent,
`0:N` for Polynomial). Defined only by classes that form a multiplication ring;
enables `*`, `kron`, `coeff`, and the para-adjoint.
"""
function powers end

"""
    nbasis(c::FunctionClass) -> Int

Number of basis weights (= number of coefficient blocks the class expects).
Falls back to `length(powers(c))` for power-window classes.
"""
nbasis(c::FunctionClass) = length(powers(c))

# class-combination rule for the matrix product / kron (windows add). Each ring
# class pair defines its own method in classes/; the core's `*`/`kron` call this.
"""
    _prodclass(a::FunctionClass, b::FunctionClass) -> FunctionClass

The class of a product `A*B` of parameterized matrices (power windows add).
Defined per ring-class pair in `classes/`.
"""
function _prodclass end

# different ring classes have no common single-parameter product (they depend on
# different parameters) — the same-class methods in classes/ are more specific and win.
function _prodclass(a::RingClass, b::RingClass)
    return error(
        "cannot combine different ring classes ($(a) × $(b)); they parameterize over " *
        "different variables — use a `ProductClass` for a genuinely multi-parameter object",
    )
end
