# para_lq: the LQ mirror of para_qr (factors as ParaMatrices). A = L·Q with L the
# EXACT spectral factor of A·para(A) and Q row-para-unitary. Exact-identity oracles
# on random wide/square A (no special structure, no cherry-picked sampling).

# full-row-rank A: an invertible left m×m block makes A(θ) full row rank ∀θ
function _fullrowrank(m, n, seed)
    left =
        paraeye(m, ComplexF64, Laurent(-1, 1)) +
        contractive(randpm(m, Laurent(-1, 1); seed=seed); bound=0.3, nsample=64)
    m == n && return left
    right = contractive(
        randpm(m, n - m, Laurent(-1, 1); seed=seed + 5); bound=0.3, nsample=64
    )
    return hcat(left, right)
end

@testset "para_lq: A = L·Q with exact spectral L and row-isometry Q (random A)" begin
    for (m, n) in ((2, 2), (3, 3), (4, 4), (2, 4), (3, 5)), seed in SEEDS
        A = _fullrowrank(m, n, seed)
        F = para_lq(A; N=32, order=40)
        @test size(F.L) == (m, m) && size(F.Q) == (m, n)
        @test F.residual < 1e-9                                        # ‖A − LQ‖ ≈ machine ε
        @test F.isometry < 1e-9                                        # ‖Q para(Q) − I‖ ≈ machine ε
        for θ in RNG_PTS
            @test A(θ) ≈ F.L(θ) * F.Q(θ) atol = 1e-9                   # reconstruction ∀θ
            @test F.Q(θ) * F.Q(θ)' ≈ I atol = 1e-9                     # Q is a row isometry ∀θ
            @test (F.L * para(F.L))(θ) ≈ (A * para(A))(θ) atol = 1e-9  # L L̃ = A Ã (L exact)
        end
    end
end

@testset "para_lq errors honestly on row-rank-deficient input" begin
    Az = ParaMatrix([ComplexF64[1 0; 0 1], ComplexF64[0 0; 0 -1]], Analytic(1))  # row 2 → 0 at θ=0
    @test_throws ErrorException para_lq(Az)
end
