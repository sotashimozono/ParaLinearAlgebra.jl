# tests src/solver/spectral.jl — spectral_factor (Bauer), para_gram, leading_eigen.

@testset "spectral_factor: G = M·para(M) (Bauer)" begin
    for d in (2, 3), seed in SEEDS
        M0 = randpm(d, Analytic(1); seed=seed)
        G = para(M0) * M0 + paraeye(d, ComplexF64, Laurent(-1, 1))     # PD para-Hermitian
        @test ishermitian(G) && ispositive(G)
        M = spectral_factor(G; N=32)
        for θ in (0.1, 0.4, 0.7, 0.95)
            @test M(θ) * para(M)(θ) ≈ G(θ) atol = 1e-6
        end
    end
end

@testset "spectral_factor guards" begin
    M0 = randpm(2, Analytic(3); seed=1)
    G = para(M0) * M0 + paraeye(2, ComplexF64, Laurent(-3, 3))
    @test_throws ArgumentError spectral_factor(G; N=2)                  # N < L = 3
    @test_throws ErrorException spectral_factor(randpm(2, Analytic(2); seed=1))  # asymmetric window
end

@testset "para_gram == para(A)·A" begin
    for seed in SEEDS
        A = randpm(3, Laurent(-1, 1); seed=seed)
        for θ in RNG_PTS
            @test para_gram(A)(θ) ≈ (para(A) * A)(θ) atol = 1e-12
        end
    end
end

@testset "leading_eigen picks the largest-|λ| per θ" begin
    for seed in SEEDS
        E = randpm(3, Laurent(-1, 1); seed=seed)
        _, λs, _ = leading_eigen(E; nsample=8)
        @test all(
            abs(λs[i]) ≈ maximum(abs, eigvals(Matrix(E(t)))) for
            (i, t) in enumerate(_circle_pts(8))
        )
    end
end
