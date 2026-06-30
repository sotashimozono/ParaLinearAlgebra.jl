# Fejér–Riesz / spectral-factorization oracle for solver/spectral.jl.
#
# Build G = M0·para(M0) from a KNOWN outer (minimum-phase) factor M0, then check
# `spectral_factor(G)` recovers a factor with the SAME modulus — the Fejér–Riesz
# invariant, fixed by G and independent of the algorithm — and that it reconstructs
# G. The factor is unique only up to a constant right-unitary, so the independent
# quantities are |M| (scalar) / |det M| (matrix) and the reconstruction, NOT M==M0.

@testset "Fejér–Riesz scalar: |spectral factor| = |known outer factor|" begin
    # M0(z) = 1 + a z with |a| < 1 has its only zero at |z| = 1/|a| > 1, so it is
    # outer; G(θ) = |1 + a e^{2πiθ}|². Bauer must return u·M0 (|u| = 1).
    for a in (0.5, 0.3im, -0.4, 0.6 * cispi(0.3))
        M0 = ParaMatrix([fill(ComplexF64(1), 1, 1), fill(ComplexF64(a), 1, 1)], Analytic(1))
        G = M0 * para(M0)
        @test isparahermitian(G) && ispositive(G)
        M = spectral_factor(G; N=32)
        @test M.class == Analytic(1)
        for θ in RNG_PTS
            @test abs(M(θ)[1, 1]) ≈ abs(1 + a * cispi(2θ)) atol = 1e-7    # modulus = |M0|  (independent)
            @test (M*para(M))(θ)[1, 1] ≈ G(θ)[1, 1] atol = 1e-7          # reconstruction G = M·para(M)
        end
    end
end

@testset "Fejér–Riesz matrix: reconstruction + |det| invariant" begin
    # M0 = I + B with ‖B(θ)‖ ≤ 0.5 ⇒ σ_min(M0(θ)) ≥ 0.5 > 0, so M0 is outer and G is PD.
    for d in (2, 3), L in (1, 2), seed in SEEDS
        M0 =
            paraeye(d, ComplexF64, Analytic(L)) +
            contractive(randpm(d, Analytic(L); seed=seed); bound=0.5, nsample=64)
        G = M0 * para(M0)
        @test ispositive(G)
        M = spectral_factor(G; N=32)
        @test M.class == Analytic(L)
        for θ in RNG_PTS
            @test (M*para(M))(θ) ≈ G(θ) atol = 1e-6                       # G = M·para(M)
            @test abs(det(M(θ))) ≈ abs(det(M0(θ))) atol = 1e-6          # |det| = √det G  (independent)
        end
    end
end
