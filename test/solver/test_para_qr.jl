# para_qr: a genuine parameterized → parameterized QR (factors returned AS
# ParaMatrices, not sampled). A = Q·R with R the EXACT spectral factor of the Gram
# and Q para-unitary. Oracles are exact algebraic identities on RANDOM A (no
# special structure, no cherry-picked sampling) — and they hold to ~machine ε.

# full-column-rank A: an invertible top n×n block makes A(θ) full rank ∀θ
function _fullrank(m, n, seed)
    top =
        paraeye(n, ComplexF64, Laurent(-1, 1)) +
        contractive(randpm(n, Laurent(-1, 1); seed=seed); bound=0.3, nsample=64)
    m == n && return top
    bot = contractive(
        randpm(m - n, n, Laurent(-1, 1); seed=seed + 5); bound=0.3, nsample=64
    )
    return vcat(top, bot)
end

@testset "para_qr: A = Q·R with exact spectral R and para-unitary Q (random A)" begin
    for (m, n) in ((2, 2), (3, 3), (4, 4), (4, 2), (5, 3)), seed in SEEDS
        A = _fullrank(m, n, seed)
        F = para_qr(A; N=32, order=40)
        @test F.Q isa ParaMatrix && F.R isa ParaMatrix
        @test size(F.Q) == (m, n) && size(F.R) == (n, n)
        @test F.residual < 1e-9                                        # ‖A − QR‖ ≈ machine ε
        @test F.isometry < 1e-9                                        # ‖para(Q)Q − I‖ ≈ machine ε
        for θ in RNG_PTS
            @test A(θ) ≈ F.Q(θ) * F.R(θ) atol = 1e-9                   # reconstruction ∀θ
            @test F.Q(θ)' * F.Q(θ) ≈ I atol = 1e-9                     # Q is a column isometry ∀θ
            @test (para(F.R) * F.R)(θ) ≈ (para(A) * A)(θ) atol = 1e-9  # R̃R = ÃA (R exact)
        end
    end
end

@testset "para_qr of a para-unitary A: Gram = I ⇒ R unitary, Q recovers A" begin
    D = ParaMatrix([ComplexF64[1 0; 0 0], ComplexF64[0 0; 0 1]], Analytic(1))  # diag(1, z)
    F = para_qr(D; N=16, order=20)
    @test F.residual < 1e-9 && F.isometry < 1e-9
    for θ in RNG_PTS
        @test (para(F.R) * F.R)(θ) ≈ I atol = 1e-9                     # para(A)A = I ⇒ R̃R = I
        @test D(θ) ≈ F.Q(θ) * F.R(θ) atol = 1e-9
    end
end

@testset "para_qr errors honestly on rank-deficient input" begin
    Az = ParaMatrix([ComplexF64[1 0; 0 1], ComplexF64[0 0; 0 -1]], Analytic(1))  # diag(1, 1−z)
    @test Az(0.0) ≈ ComplexF64[1 0; 0 0]                               # rank 1 at θ = 0
    @test_throws ErrorException para_qr(Az)                            # Gram not PD ⇒ no QR
end

@testset "multivariate (ProductClass) parameterized factorization is gated" begin
    # ≥2 parameters: parameterized factorization is unavailable by design (hard/open);
    # the methods must error clearly. (The SAMPLED eigen/svd DO support ProductClass.)
    pc = ProductClass(Laurent(-1, 1), Laurent(-1, 1))
    A = ParaMatrix([randn(MersenneTwister(i), ComplexF64, 2, 2) for i in 1:nbasis(pc)], pc)
    @test_throws ErrorException spectral_factor(A)
    @test_throws ErrorException para_qr(A)
    @test_throws ErrorException para_lq(A)
    @test_throws ErrorException para_svd(A)
    @test_throws ErrorException para_eigen(A)
end
