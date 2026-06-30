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
[`basis_deriv`](@ref) and [`powers`](@ref). The core is generic over this
interface — individual classes are defined separately in `classes/`.
"""
abstract type FunctionClass end

"""
    basis(c::FunctionClass, p) -> AbstractVector

The weights `w(p)` of class `c` at parameter `p`.
"""
function basis end

"""
    basis_deriv(c::FunctionClass, p) -> AbstractVector

The parameter derivative `dw/dp` of the weights — the primitive behind
parameter-derivatives of any object built on `c` (Berry/quantum-geometry).
Optional; defined by classes that are differentiable in `p`.
"""
function basis_deriv end

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
