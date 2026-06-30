# tests src/utils/utils.jl — circle sweep, rank diagnostics, first-order descent.

@testset "on_circle bridges any pointwise routine" begin
    A = randpm(3, Laurent(-1, 1); seed=1)
    ts, vals = on_circle(eigvals, A; nsample=8)
    @test ts == _circle_pts(8)
    @test all(vals[i] ≈ eigvals(Matrix(A(ts[i]))) for i in eachindex(ts))
    ts2, trs = on_circle(tr, A; nsample=8)
    @test all(trs[i] ≈ tr(Matrix(A(ts2[i]))) for i in eachindex(ts2))
end

@testset "rank / rank_profile / numerical_rank detect the rank drop" begin
    rk = ParaMatrix([ComplexF64[1 0; 0 0], ComplexF64[0 1; 0 0]], Laurent(0, 1))  # rank 1 ∀θ
    @test rank_profile(rk; nsample=32, tol=1e-9) == (1, 1, 0.0)
    @test numerical_rank(rk; nsample=16) == LinearAlgebra.rank(rk; nsample=16) == 1
    # full-rank random matrix has full numerical rank
    for seed in SEEDS
        A = randpm(4, Laurent(-1, 1); seed=seed)
        @test numerical_rank(A; nsample=16) == 4
    end
end

@testset "optimize! decreases a convex coefficient loss" begin
    for seed in SEEDS
        B = ParaMatrix(
            [randn(MersenneTwister(seed + i), 2, 2) for i in 1:3], Laurent(-1, 1)
        )
        L(M) = sum(norm(c)^2 for c in M.coeffs)
        g(M) = 2 .* M.coeffs
        _, hist = optimize!(B, g; steps=50, lr=0.05, loss=L)
        @test hist[end] < hist[1] && issorted(hist; rev=true)
    end
end
