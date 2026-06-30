# utils/utils.jl — circle sampling, rank diagnostics, first-order optimization.

# the uniform sampling grid of the circle θ ∈ [0,1)
_circle(n::Int) = range(0, 1; length=n + 1)[1:n]

"""
    on_circle(f, A::ParaMatrix; nsample=128) -> (ts, ys)

Sample `f(A(θ))` over the circle grid `ts`. This is the generic bridge for any
pointwise LinearAlgebra routine — e.g. `on_circle(eigvals, A)`,
`on_circle(svdvals, A)`, `on_circle(qr, A)` give the eigenvalue / singular-value
/ QR *functions* over the torus.
"""
function on_circle(f, A::ParaMatrix; nsample::Int=128)
    ts = _circle(nsample)
    return ts, [f(Matrix(A(t))) for t in ts]
end

"""
    rank(A::ParaMatrix; nsample=128, tol=1e-9) -> Int

The maximum pointwise numerical rank over the circle (the bond-dimension
diagnostic of a transfer/cocycle ParaMatrix).
"""
function LinearAlgebra.rank(A::ParaMatrix; nsample::Int=128, tol::Real=1e-9)
    return maximum(count(>(tol), svdvals(Matrix(A(t)))) for t in _circle(nsample))
end

"""
    numerical_rank(A::ParaMatrix; nsample=128, tol=1e-9) -> Int

Alias for [`rank`](@ref) on a `ParaMatrix` (max pointwise numerical rank over the
circle) — the bond-dimension diagnostic of a transfer/cocycle ParaMatrix.
"""
function numerical_rank(A::ParaMatrix; nsample::Int=128, tol::Real=1e-9)
    return rank(A; nsample=nsample, tol=tol)
end

"""
    rank_profile(A::ParaMatrix; nsample=256, tol=1e-9) -> (minrank, maxrank, θ_of_min)

Rank-drop detection over the circle (gauge / rank-deficiency diagnostic).
"""
function rank_profile(A::ParaMatrix; nsample::Int=256, tol::Real=1e-9)
    ts = _circle(nsample)
    ranks = [count(>(tol), svdvals(Matrix(A(t)))) for t in ts]
    return (minimum(ranks), maximum(ranks), ts[argmin(ranks)])
end

"""
    optimize!(A::ParaMatrix, grad; steps=200, lr=0.05, loss=nothing) -> (A, history)

First-order descent on the coefficients. `grad(A) -> Vector` supplies the
coefficient cotangents (from the ChainRulesCore rrules, Zygote, Mooncake, or
finite differences). Mutates `A.coeffs` in place.
"""
function optimize!(A::ParaMatrix, grad; steps::Int=200, lr::Real=0.05, loss=nothing)
    history = Float64[]
    for _ in 1:steps
        g = grad(A)
        for i in eachindex(A.coeffs)
            @. A.coeffs[i] -= lr * g[i]
        end
        loss === nothing || push!(history, float(real(loss(A))))
    end
    return A, history
end
