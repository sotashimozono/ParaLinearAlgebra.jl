# solver/equations.jl — parameterized matrix equations / linear solves.
# Return shapes follow KrylovKit: a solution plus an `info` NamedTuple.

# discrete Lyapunov / Stein solve  X = A X A† + Q  (vectorized (I − Ā⊗A) x = vec Q).
# Bounded solution exists only for ρ(A) < 1, which is checked (otherwise the result
# is meaningless / the system is singular).
function _stein_solve(A::AbstractMatrix, Q::AbstractMatrix)
    ρ = maximum(abs, eigvals(A))
    ρ < 1 || throw(
        ArgumentError(
            "lyapd/_stein_solve: spectral radius ρ(A)=$(round(ρ; digits=6)) ≥ 1; the " *
            "discrete-Lyapunov (Stein) equation has no bounded solution",
        ),
    )
    n = size(A, 1)
    X = reshape((I - kron(conj(A), A)) \ vec(Q), n, n)
    return (X + X') / 2
end

"""
    lyapd(A, Q; nsample=128) -> (ts, Xs)

Pointwise discrete-Lyapunov (Stein) fixed point `X(θ) = A(θ) X(θ) A(θ)† + Q(θ)`
on the circle. Requires `ρ(A(θ)) < 1` at every sample (a clear error is raised
otherwise).
"""
function lyapd(A::ParaMatrix{T,S,<:Laurent}, Q::ParaMatrix; nsample::Int=128) where {T,S}
    ts = _circle(nsample)
    Xs = [_stein_solve(Matrix(A(t)), Matrix(Q(t))) for t in ts]
    return ts, Xs
end

"""
    cocycle_exponent(E, p, q; θ0=0.0) -> Real

The cocycle (dynamical-shift) Lyapunov exponent: per-site log growth of the
transfer cocycle under `θ → θ + p/q`, accumulated with rescaling for stability.
`p/q = Fₙ/Fₙ₊₁ → 1/φ` recovers the irrational (phason) environment rate.
"""
function cocycle_exponent(
    E::ParaMatrix{T,S,<:Laurent}, p::Int, q::Int; θ0::Real=0.0
) where {T,S}
    α = p / q
    n = size(E, 1)
    M = Matrix{complex(float(T))}(I, n, n)
    logscale = 0.0
    for k in 0:(q - 1)
        M = Matrix(E(mod(θ0 + k * α, 1.0))) * M
        nrm = norm(M)
        iszero(nrm) && throw(
            ArgumentError(
                "cocycle_exponent: transfer cocycle collapsed to zero at step k=$k " *
                "(θ=$(round(mod(θ0 + k * α, 1.0); digits=6))); exponent is -∞/undefined",
            ),
        )
        M ./= nrm
        logscale += log(nrm)
    end
    return (logscale + log(maximum(abs, eigvals(M)))) / q
end

"""
    para_solve(A, b; order=8, nsample=0) -> (x, info)

Solve `A(θ) x(θ) = b(θ)` by per-θ dense solve + a `Laurent(order)` least-squares
fit, KrylovKit-style: returns the fitted `x::ParaMatrix` and
`info = (; converged, residual)` where `residual = max_θ ‖A(θ)x(θ) − b(θ)‖`.
A large residual means `x` is genuinely rational and needs a higher `order`.
"""
function para_solve(
    A::ParaMatrix{T,S,<:Laurent},
    b::ParaMatrix;
    order::Int=8,
    nsample::Int=0,
    tol::Real=1e-8,
) where {T,S}
    CT = complex(float(T))
    L = order
    ns = nsample == 0 ? 4L + 8 : nsample
    ts = _circle(ns)
    xs = [Matrix{CT}(A(t)) \ Matrix{CT}(b(t)) for t in ts]
    W = [cispi(2 * k * t) for t in ts, k in (-L):L]
    coef = W \ reduce(vcat, (permutedims(vec(x)) for x in xs))
    d1, d2 = size(xs[1])
    x = ParaMatrix([Matrix(reshape(coef[m, :], d1, d2)) for m in 1:(2L + 1)], Laurent(L))
    fine = _circle(4ns)
    resid = maximum(opnorm(Matrix(A(t)) * Matrix(x(t)) - Matrix(b(t))) for t in fine)
    return x, (; converged=resid ≤ tol, residual=resid)
end

"""
    A \\ b

The fitted Laurent solution of `A(θ) x(θ) = b(θ)` (see [`para_solve`](@ref)). Warns
if the fit does not converge (the true solution is rational / needs a higher order);
use [`para_solve`](@ref) directly to inspect `info.residual`.
"""
function Base.:\(A::ParaMatrix{T,S,<:Laurent}, b::ParaMatrix) where {T,S}
    x, info = para_solve(A, b)
    info.converged ||
        @warn "A\\b: Laurent fit did not converge (residual=$(info.residual)); the solution is rational — raise `order` via para_solve." maxlog =
            3
    return x
end
