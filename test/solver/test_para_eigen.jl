# para_eigen (PEVD): H(θ) ≈ U(θ)·D(θ)·U(θ)' for para-Hermitian H, eigenvalue
# functions D real (and possibly NEGATIVE — unlike singular values). Necessarily
# approximate (same Berry/Zak obstruction as para_svd). Oracle: recover the
# gauge-invariant eigenvalue functions + reconstruct, from zero-Zak constructions
# (~machine ε); the obstructed (Zak≠0) case is correctly flagged while D survives.

function _constunitary(d, seed)
    return ParaMatrix(
        [Matrix(qr(randn(MersenneTwister(seed), ComplexF64, d, d)).Q)], Laurent(0, 0)
    )
end

# real diagonal D(θ) = diag(cᵢ + aᵢ·cos2πθ), separated (cs may be negative)
function _realdiag(cs, as)
    z0 = Matrix{ComplexF64}(Diagonal(ComplexF64.(cs)))
    zpm = Matrix{ComplexF64}(Diagonal(ComplexF64.(as ./ 2)))
    return ParaMatrix([zpm, z0, zpm], Laurent(-1, 1))
end

@testset "para_eigen: eigenvalue functions + reconstruction (constant eigenvectors)" begin
    cs = [2.0, 0.5, -1.5]                                   # separated, includes a NEGATIVE band
    as = [0.3, 0.2, 0.25]
    for d in (2, 3), seed in SEEDS
        U0 = _constunitary(d, seed)
        H = U0 * _realdiag(cs[1:d], as[1:d]) * para(U0)     # para-Hermitian
        @test isparahermitian(H)
        F = para_eigen(H; order=8)
        @test F.residual < 1e-7
        @test maximum(abs, F.winding) < 1e-6
        for θ in RNG_PTS
            @test H(θ) ≈ F.U(θ) * F.D(θ) * F.U(θ)' atol = 1e-6                       # H = UDU'
            @test sort(real(diag(F.D(θ)))) ≈ eigvals(Hermitian(Matrix(H(θ)))) atol = 1e-6  # bands
            @test F.U(θ)' * F.U(θ) ≈ I atol = 1e-6                                  # U para-unitary
            @test maximum(abs, imag(diag(F.D(θ)))) < 1e-7                           # eigenvalues real
        end
    end
end

@testset "para_eigen: θ-dependent eigenvectors (rotation, real-symmetric H)" begin
    c = ComplexF64
    Id = c[1 0; 0 1]
    J = c[0 -1; 1 0]
    R = ParaMatrix([0.5Id + 0.5im * J, zeros(c, 2, 2), 0.5Id - 0.5im * J], Laurent(-1, 1))
    H = R * _realdiag([2.0, -1.0], [0.3, 0.2]) * para(R)
    @test isparahermitian(H)
    F = para_eigen(H; order=6)
    @test F.residual < 1e-9
    for θ in RNG_PTS
        @test H(θ) ≈ F.U(θ) * F.D(θ) * F.U(θ)' atol = 1e-8
        @test sort(real(diag(F.D(θ)))) ≈ eigvals(Hermitian(Matrix(H(θ)))) atol = 1e-8
    end
end

@testset "para_eigen reports the obstruction (Zak≠0) while D is still recovered" begin
    c = ComplexF64
    Q1 = Matrix(qr(randn(MersenneTwister(7), c, 2, 2)).Q)
    Q2 = Matrix(qr(randn(MersenneTwister(8), c, 2, 2)).Q)
    Dz = ParaMatrix([c[1 0; 0 0], c[0 0; 0 1]], Analytic(1))                # diag(1, z): winding eigvecs
    U0 = ParaMatrix([Q1], Analytic(0)) * Dz * ParaMatrix([Q2], Analytic(0))
    H = U0 * _realdiag([2.0, -1.0], [0.0, 0.0]) * para(U0)                  # bands = {2,−1}, constant
    @test isparahermitian(H)
    F = @test_logs (:warn,) match_mode = :any para_eigen(H; order=8)
    @test F.residual > 1e-3
    @test maximum(abs, F.winding) > 1e-2
    for θ in RNG_PTS
        @test sort(real(diag(F.D(θ)))) ≈ eigvals(Hermitian(Matrix(H(θ)))) atol = 1e-6  # D recovered
    end
end

@testset "para_eigen warns on non-para-Hermitian input" begin
    A = randpm(2, Laurent(-1, 1); seed=1)
    @test !isparahermitian(A)
    @test_logs (:warn,) match_mode = :any para_eigen(A; order=6)
end

# order-2 real separated diagonal: dᵢ(θ) = cᵢ + 0.2cos2πθ + 0.15cos4πθ
function _realdiag2(cs)
    d = length(cs)
    z0 = Matrix{ComplexF64}(Diagonal(ComplexF64.(cs)))
    z1 = Matrix{ComplexF64}(Diagonal(fill(ComplexF64(0.1), d)))
    z2 = Matrix{ComplexF64}(Diagonal(fill(ComplexF64(0.075), d)))
    return ParaMatrix([z2, z1, z0, z1, z2], Laurent(-2, 2))
end

@testset "para_eigen refinements: mingap diagnostic + adaptive order" begin
    U0 = _constunitary(3, 5)
    H = U0 * _realdiag([2.0, 0.5, -1.5], [0.2, 0.2, 0.2]) * para(U0)
    F = para_eigen(H; order=8)
    @test F.mingap > 0.5                                  # well-separated bands
    @test F.order == 8                                    # fixed order when tol=0

    Hhi = U0 * _realdiag2([2.0, 0.0, -2.0]) * para(U0)    # bands need order ≥ 2
    @test para_eigen(Hhi; order=1).residual > 1e-6
    F2 = para_eigen(Hhi; order=1, tol=1e-9, maxorder=20)
    @test F2.residual ≤ 1e-9
    @test F2.order > 1

    Hx = U0 * _realdiag([1.0, 1.0001, 5.0], [0.0, 0.0, 0.0]) * para(U0)
    Fx = @test_logs (:warn,) match_mode = :any para_eigen(Hx; order=8, gaptol=1e-2)
    @test Fx.mingap < 1e-2
end

@testset "para_eigvals: differentiable eigenvalue functions (counterpart of para_eigen)" begin
    cs = [2.0, 0.5, -1.5]
    as = [0.3, 0.2, 0.25]
    for d in (2, 3), seed in SEEDS
        U0 = _constunitary(d, seed)
        H = U0 * _realdiag(cs[1:d], as[1:d]) * para(U0)
        D = para_eigvals(H; order=6)
        for θ in RNG_PTS
            @test sort(real(diag(D(θ)))) ≈ eigvals(Hermitian(Matrix(H(θ)))) atol = 1e-6  # = eigvals
        end
    end
end
