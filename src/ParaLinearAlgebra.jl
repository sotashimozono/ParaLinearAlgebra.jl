"""
    ParaLinearAlgebra

A generic, backend-agnostic algebra for **parameterized matrices**
`A(p) = Σ_k basis(class, p)_k · coeffs_k`, with automatic differentiation in the
coefficients and parameter-derivatives for quantum-geometry.

The package is layered so the abstract backend never depends on any specific
function type:

- `core/`    — the [`FunctionClass`](@ref) interface and the [`ParaMatrix`](@ref)
               type with its class-agnostic algebra (`evaluate`, the callable
               `A(p)`, ring ops, AD rrules).
- `classes/` — concrete function types as plug-ins: [`Fourier`](@ref) (ansatz),
               [`Laurent`](@ref)/[`Analytic`](@ref) (ring + para-adjoint),
               [`Polynomial`](@ref), [`ProductClass`](@ref).
- `solver/`  — factorizations (`eigen`/`svd`/`qr`/`lq`/`lu`/`pinv` dispatch on
               `ParaMatrix`) plus circle/`Laurent`-parametric algorithms
               ([`spectral_factor`](@ref), [`para_gram`](@ref),
               [`leading_eigen`](@ref), `lyapd`,
               [`cocycle_exponent`](@ref), [`para_solve`](@ref) — these require a
               `Laurent` class).
- `utils/`   — [`on_circle`](@ref), `rank`/[`rank_profile`](@ref), [`optimize!`](@ref).

UI follows the Julia ecosystem: a `ParaMatrix` is **callable** (`A(p)` → a dense
matrix, so all of `LinearAlgebra` works pointwise); in-class operations reuse the
standard verbs (`det`, `inv`, `tr`, `norm`, `kron`, `adjoint`/`'`, `ishermitian`,
`isposdef`, `\\`); iterative-style solvers return `(…, info)` à la KrylovKit.
"""
module ParaLinearAlgebra

using LinearAlgebra
using ChainRulesCore
using ChainRulesCore: rrule, NoTangent, ZeroTangent, AbstractZero, Tangent, unthunk
# extend the ecosystem's discrete-Lyapunov solver (O(n³) Schur) rather than
# reimplement it — `lyapd(::ParaMatrix, ::ParaMatrix)` adds the per-θ method.
import MatrixEquations: lyapd

# core: abstract interface + central type + AD
include("core/function_class.jl")
include("core/paramatrix.jl")
include("core/chainrules.jl")

# classes: concrete function-type plug-ins
include("classes/fourier.jl")
include("classes/laurent.jl")
include("classes/polynomial.jl")
include("classes/product.jl")
include("core/blockparamatrix.jl")

# solver: algorithms on a ParaMatrix
include("solver/spectral.jl")
include("solver/factorizations.jl")
include("solver/equations.jl")

# utils
include("utils/utils.jl")

# ---- exports ----
# core
export FunctionClass,
    RingClass, basis, basis_deriv, basis_gram, basis_integral, nbasis, powers
export ParaMatrix, evaluate, evaluate_deriv, coefficients, function_class, coeff, nterms
export BlockParaMatrix, nparams
export integral
export paraeye, ⊗, ⊕, directsum
# classes
export Fourier, Laurent, Analytic, Polynomial, ProductClass
# para-structure
export para, paraconj, parahermitianpart, isparahermitian, isparaunitary, ispositive
# solver — factorizations dispatch on the STANDARD verbs (eigen/svd/qr/lq/lu/
# eigvals/svdvals/pinv) for ParaMatrix; these are the returned objects + polar.
export ParaEigen, ParaSVD, ParaQR, ParaLQ, ParaLU, ParaPolar, polar, numerical_rank
# solver — spectral / equations
export para_gram, spectral_factor, para_qr, para_lq, para_svd, para_eigen
export para_svdvals, para_eigvals
export leading_eigen, lyapd, cocycle_exponent, para_solve
# utils
export on_circle, rank_profile, optimize!

end # module ParaLinearAlgebra
