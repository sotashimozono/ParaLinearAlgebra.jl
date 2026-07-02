# classes/function_basis.jl — a generic ANSATZ class from an explicit list of
# scalar basis functions f_k(p) (and, optionally, their derivatives).
#
# This is the open door to arbitrary parameter dependence — classical orthogonal
# polynomials, `SpecialFunctions.jl` (Bessel, Airy, …), exponentials, or any
# hand-written ansatz — without the core naming the family. Like `Fourier` it is
# NOT a ring class (arbitrary functions have no multiplication window), so it
# supports construct / `evaluate` / `∂_p` / AD / (optionally) the L² inner
# product, but not the ParaMatrix product `*`/`kron`/`para`.

"""
    FunctionBasis(funcs; derivs=nothing, gram=nothing, integral=nothing,
                  interval=nothing, weight=nothing, quadnodes=nothing,
                  label="FunctionBasis") <: FunctionClass

An ansatz [`FunctionClass`](@ref) whose weights are an explicit list of scalar
functions: `basis(c, p) = [f(p) for f in funcs]`, so
`A(p) = Σ_k funcs[k](p) · coeffs[k]`. The escape hatch for parameter dependence
the built-in classes do not cover — classical orthogonal polynomials,
`SpecialFunctions.jl` (Bessel/Airy/…), exponentials, or a hand-written ansatz.

`funcs` is any indexable collection of callables `p -> Number` (a `Vector` or a
`Tuple`).

Optional data unlock more of the algebra:
- `derivs` — a same-length collection of callables `p -> Number` giving `fₖ'(p)`;
  enables [`evaluate_deriv`](@ref) / [`basis_deriv`](@ref) (parameter sensitivities).
- `gram` / `integral` — the `n×n` L² Gram `Mₖₗ = ∫ conj(fₖ) fₗ` and the vector
  `∫ fₖ`; enable [`dot`](@ref) / [`norm`](@ref) / [`integral`](@ref).
- `interval=(a,b)` — a convenience that fills `gram`/`integral` by Gauss–Legendre
  quadrature of `conj(fₖ) fₗ · weight` over `[a,b]` (`weight` defaults to `1`,
  `quadnodes` to `max(8, 2·nbasis)` nodes; exact for a polynomial basis of degree
  `< quadnodes`). `norm(A)` is then the honest `L²([a,b], weight)` norm.

Being an ansatz class, `FunctionBasis` has no `*`/`kron`/`para` (use `Laurent`
for the circle ring or `Polynomial` for the monomial ring). Binary operations
(`A + B`, …) compare classes for equality, and a `FunctionBasis` holds its
functions by reference, so build the operands from the **same** `FunctionBasis`
object.
"""
struct FunctionBasis{FS,DS,G,V} <: FunctionClass
    funcs::FS
    derivs::DS
    gram::G
    integral::V
    label::String
    function FunctionBasis(funcs, derivs, gram, integral, label::AbstractString)
        n = length(funcs)
        n ≥ 1 || throw(ArgumentError("FunctionBasis needs at least one basis function"))
        derivs === nothing ||
            length(derivs) == n ||
            throw(
                ArgumentError(
                    "derivs must have length $n (one per basis function); got $(length(derivs))",
                ),
            )
        gram === nothing ||
            size(gram) == (n, n) ||
            throw(ArgumentError("gram must be $n×$n; got $(size(gram))"))
        integral === nothing ||
            length(integral) == n ||
            throw(ArgumentError("integral must have length $n; got $(length(integral))"))
        return new{typeof(funcs),typeof(derivs),typeof(gram),typeof(integral)}(
            funcs, derivs, gram, integral, String(label)
        )
    end
end

function FunctionBasis(
    funcs;
    derivs=nothing,
    gram=nothing,
    integral=nothing,
    interval=nothing,
    weight=nothing,
    quadnodes=nothing,
    label::AbstractString="FunctionBasis",
)
    if interval !== nothing
        (gram === nothing && integral === nothing) || throw(
            ArgumentError(
                "pass EITHER `interval` (quadrature) OR explicit `gram`/`integral`, not both",
            ),
        )
        a, b = float(first(interval)), float(last(interval))
        a < b || throw(ArgumentError("interval must be (a,b) with a < b; got ($a, $b)"))
        n = length(funcs)
        m = quadnodes === nothing ? max(8, 2n) : Int(quadnodes)
        xs, ws = _gauss_legendre(m, a, b)
        wf = weight === nothing ? (_ -> 1.0) : weight
        # weights of every basis function at every node: W[i] = [f(xs[i]) for f in funcs]
        W = [[f(x) for f in funcs] for x in xs]
        gram = [
            sum(ws[i] * wf(xs[i]) * conj(W[i][k]) * W[i][l] for i in 1:m) for
            k in 1:n, l in 1:n
        ]
        integral = [sum(ws[i] * wf(xs[i]) * W[i][k] for i in 1:m) for k in 1:n]
    end
    return FunctionBasis(funcs, derivs, gram, integral, label)
end

nbasis(c::FunctionBasis) = length(c.funcs)
basis(c::FunctionBasis, p) = [f(p) for f in c.funcs]

function basis_deriv(c::FunctionBasis, p)
    c.derivs === nothing && throw(
        ArgumentError(
            "this FunctionBasis carries no derivatives; construct it with `derivs=` to enable ∂_p",
        ),
    )
    return [g(p) for g in c.derivs]
end

function basis_gram(c::FunctionBasis)
    c.gram === nothing && throw(
        ArgumentError(
            "this FunctionBasis carries no Gram; construct it with `gram=` or `interval=` to enable dot/norm",
        ),
    )
    return c.gram
end

function basis_integral(c::FunctionBasis)
    c.integral === nothing && throw(
        ArgumentError(
            "this FunctionBasis carries no integral vector; construct it with `integral=` or `interval=` to enable integral",
        ),
    )
    return c.integral
end

# Gauss–Legendre nodes/weights on [a,b] via Golub–Welsch: the nodes are the
# eigenvalues of the (Legendre) Jacobi matrix — a symmetric tridiagonal with
# βₖ = k/√(4k²−1) — and the weights are μ₀·(first eigenvector component)², μ₀=2.
# Uses only LinearAlgebra (already a dependency); exact for polynomials of degree
# ≤ 2m−1. Returns nodes/weights mapped affinely to [a,b].
function _gauss_legendre(m::Integer, a::Real, b::Real)
    m ≥ 1 || throw(ArgumentError("Gauss–Legendre needs m ≥ 1 nodes; got $m"))
    if m == 1
        return [(a + b) / 2], [float(b - a)]
    end
    β = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(m - 1)]
    E = eigen(SymTridiagonal(zeros(m), β))
    x = E.values
    w = 2 .* (E.vectors[1, :] .^ 2)
    h = (b - a) / 2
    return h .* x .+ (a + b) / 2, h .* w
end

# ---- extension hooks (methods added by weak-dependency package extensions) ----

"""
    polynomial_basis(P, N; interval=nothing) -> FunctionClass

A [`FunctionClass`](@ref) whose basis is the first `N+1` basis polynomials of the
`Polynomials.jl` family `P` — e.g. `ChebyshevT`, `Polynomials.Polynomial`, or a
`SpecialPolynomials.jl` family such as `Legendre`/`Hermite`/`Laguerre`. Exact
derivatives and the L² Gram/integral (over the family's standard domain, or
`interval=(a,b)`) come from Polynomials.jl. Convert the result into the monomial
ring with [`monomialize`](@ref) to gain `*`/`kron`/`^`.

Requires the Polynomials.jl extension: `using Polynomials` (add `Polynomials` to
your project; ParaLinearAlgebra loads the extension automatically).
"""
function polynomial_basis(args...; kwargs...)
    return error(
        "polynomial_basis requires the Polynomials.jl extension — add `Polynomials` to your " *
        "project and `using Polynomials` (the extension then loads automatically).",
    )
end

"""
    monomialize(A::AbstractParaMatrix) -> ParaMatrix

Re-express a ParaMatrix built on a [`polynomial_basis`](@ref) family in the
monomial [`Polynomial`](@ref) ring class by exact change of basis (values are
preserved: `monomialize(A)(x) == A(x)`), so the ring operations `*`/`kron`/`^`
become available. Pairs with [`polynomial_basis`](@ref): construct in a
well-conditioned orthogonal basis, then `monomialize` to multiply.

Requires the Polynomials.jl extension: `using Polynomials`.
"""
function monomialize(args...; kwargs...)
    return error("monomialize requires the Polynomials.jl extension — `using Polynomials`.")
end
