# para_svd: parameterized SVD with factors AS ParaMatrices. Necessarily approximate
# (singular values/vectors are analytic, generally not finite Laurent — and the
# vectors are only Laurent when the per-band Berry/Zak phase vanishes). The oracle:
# recover the GAUGE-INVARIANT singular-value functions and reconstruct A, built from
# constructions whose factors ARE clean Laurent (zero Zak phase) so it must reach
# ~machine precision. (The reported `winding`/`residual` flag the obstructed case.)

# random CONSTANT unitary (θ-independent ⇒ zero Zak phase)
function _constunitary(d, seed)
    return ParaMatrix(
        [Matrix(qr(randn(MersenneTwister(seed), ComplexF64, d, d)).Q)], Laurent(0, 0)
    )
end

# Σ(θ) = diag(cᵢ + aᵢ·cos2πθ), positive and separated
function _posdiag(cs, as)
    z0 = Matrix{ComplexF64}(Diagonal(ComplexF64.(cs)))
    zpm = Matrix{ComplexF64}(Diagonal(ComplexF64.(as ./ 2)))
    return ParaMatrix([zpm, z0, zpm], Laurent(-1, 1))
end

@testset "para_svd: σ functions + reconstruction, constant singular vectors" begin
    cs = [3.0, 2.0, 1.0]
    as = [0.25, 0.2, 0.15]
    for d in (2, 3), seed in SEEDS
        U0 = _constunitary(d, seed)
        V0 = _constunitary(d, seed + 100)
        Σ = _posdiag(cs[1:d], as[1:d])
        A = U0 * Σ * para(V0)                                  # A = U0·Σ(θ)·V0'
        F = para_svd(A; order=8)
        @test F.residual < 1e-7                                # reconstruction ≈ machine ε
        @test maximum(abs, F.winding) < 1e-6                  # zero Zak phase
        for θ in RNG_PTS
            @test A(θ) ≈ F.U(θ) * F.S(θ) * F.V(θ)' atol = 1e-6                  # A = UΣV'
            @test sort(real(diag(F.S(θ))); rev=true) ≈ svdvals(A(θ)) atol = 1e-6  # σ functions
            @test F.U(θ)' * F.U(θ) ≈ I atol = 1e-6                             # U cols orthonormal
            @test F.V(θ)' * F.V(θ) ≈ I atol = 1e-6
        end
    end
end

@testset "para_svd: θ-dependent singular vectors (rotation ⊗ SU(2), real/zero-Zak)" begin
    c = ComplexF64
    Id = c[1 0; 0 1]
    J = c[0 -1; 1 0]
    sx = c[0 1; 1 0]
    U0 = ParaMatrix([0.5Id + 0.5im * J, zeros(c, 2, 2), 0.5Id - 0.5im * J], Laurent(-1, 1))  # rotation
    V0 = ParaMatrix([0.5Id - 0.5sx, zeros(c, 2, 2), 0.5Id + 0.5sx], Laurent(-1, 1))          # SU(2)
    Σ = _posdiag([2.0, 1.0], [0.3, 0.2])
    A = U0 * Σ * para(V0)
    F = para_svd(A; order=6)
    @test F.residual < 1e-9
    for θ in RNG_PTS
        @test A(θ) ≈ F.U(θ) * F.S(θ) * F.V(θ)' atol = 1e-8
        @test sort(real(diag(F.S(θ))); rev=true) ≈ [2 + 0.3cospi(2θ), 1 + 0.2cospi(2θ)] atol =
            1e-8
    end
end

@testset "para_svd reports the obstruction (nonzero Berry phase ⇒ residual + winding)" begin
    # C1·diag(1,z)·C2 has singular vectors with nonzero Zak phase ⇒ the smooth gauge is
    # NOT periodic/Laurent: the method must flag it (large residual, nonzero winding),
    # while the singular VALUES are still recovered (gauge-invariant, here all = 1).
    c = ComplexF64
    Q1 = Matrix(qr(randn(MersenneTwister(7), c, 2, 2)).Q)
    Q2 = Matrix(qr(randn(MersenneTwister(8), c, 2, 2)).Q)
    D = ParaMatrix([c[1 0; 0 0], c[0 0; 0 1]], Analytic(1))           # diag(1, z)
    A = ParaMatrix([Q1], Analytic(0)) * D * ParaMatrix([Q2], Analytic(0))
    F = @test_logs (:warn,) match_mode = :any para_svd(A; order=8)
    @test F.residual > 1e-3                                            # honestly flagged
    @test maximum(abs, F.winding) > 1e-2
    for θ in RNG_PTS
        @test sort(real(diag(F.S(θ))); rev=true) ≈ ones(2) atol = 1e-6  # σ ≡ 1 still recovered
    end
end

# order-2 separated positive diagonal: dᵢ(θ) = cᵢ + 0.2cos2πθ + 0.15cos4πθ
function _posdiag2(cs)
    d = length(cs)
    z0 = Matrix{ComplexF64}(Diagonal(ComplexF64.(cs)))
    z1 = Matrix{ComplexF64}(Diagonal(fill(ComplexF64(0.1), d)))
    z2 = Matrix{ComplexF64}(Diagonal(fill(ComplexF64(0.075), d)))
    return ParaMatrix([z2, z1, z0, z1, z2], Laurent(-2, 2))
end

@testset "para_svd refinements: mingap diagnostic + adaptive order" begin
    U0 = _constunitary(3, 5)
    V0 = _constunitary(3, 6)
    A = U0 * _posdiag([3.0, 2.0, 1.0], [0.2, 0.2, 0.2]) * para(V0)
    F = para_svd(A; order=8)
    @test F.mingap > 0.5                                  # well-separated bands
    @test F.order == 8                                    # fixed order when tol=0

    # adaptive order: a fixed low order is poor; with tol it grows to meet it
    Ahi = U0 * _posdiag2([3.0, 2.0, 1.0]) * para(V0)      # needs order ≥ 2
    @test para_svd(Ahi; order=1).residual > 1e-6
    F2 = para_svd(Ahi; order=1, tol=1e-9, maxorder=20)
    @test F2.residual ≤ 1e-9
    @test F2.order > 1

    # near-degeneracy ⇒ small mingap + @warn
    Ax = U0 * _posdiag([1.0, 1.0001, 5.0], [0.0, 0.0, 0.0]) * para(V0)
    Fx = @test_logs (:warn,) match_mode = :any para_svd(Ax; order=8, gaptol=1e-2)
    @test Fx.mingap < 1e-2
end
