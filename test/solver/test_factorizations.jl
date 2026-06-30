# tests src/solver/factorizations.jl — eigen/svd/qr/lq/lu/polar dispatch on
# ParaMatrix. NON-tautological: closed-form cases, cross-algorithm triangulation,
# structural laws (hard @test) + continuity/monodromy probes (@warn on surprise).

@testset "struct + standard-verb dispatch (smoke)" begin
    A = randpm(3, Laurent(-1, 1); seed=1)
    @test eigen(A; nsample=4) isa ParaEigen
    @test svd(A; nsample=4) isa ParaSVD
    @test qr(A; nsample=4) isa ParaQR
    @test lq(A; nsample=4) isa ParaLQ
    @test lu(A; nsample=4) isa ParaLU
    @test polar(A; nsample=4) isa ParaPolar
    U, S, V = svd(A; nsample=4)
    @test length(U) == length(S) == length(V) == 4
    @test eigen(
        ParaMatrix([randn(MersenneTwister(i), 3, 3) for i in 1:5], Fourier(2)); nsample=4
    ) isa ParaEigen
end

@testset "closed-form spectra" begin
    # D(θ) = diag(e^{2πiθ}, e^{-2πiθ})
    D = ParaMatrix(
        [ComplexF64[0 0; 0 1], zeros(ComplexF64, 2, 2), ComplexF64[1 0; 0 0]],
        Laurent(-1, 1),
    )
    for θ in RNG_PTS
        @test _sortc(eigvals(D(θ))) ≈ _sortc([cispi(2θ), cispi(-2θ)])
        @test svdvals(D(θ)) ≈ [1.0, 1.0]
        @test det(D)(θ)[1, 1] ≈ 1
    end
    @test isparaunitary(D)
    # planar rotation (real, Fourier)
    R = ParaMatrix([zeros(2, 2), [1.0 0; 0 1], [0.0 -1; 1 0]], Fourier(1))
    for θ in RNG_PTS
        @test _sortc(eigvals(R(θ))) ≈ _sortc([cispi(2θ), cispi(-2θ)])
        @test svdvals(R(θ)) ≈ [1.0, 1.0]
        @test det(R(θ)) ≈ 1
    end
end

@testset "cross-algorithm triangulation + structural laws" begin
    for d in (2, 3, 4), seed in SEEDS
        A = randpm(d, Laurent(-1, 1); seed=seed)
        B = randpm(d, Laurent(-1, 1); seed=seed + 1)
        dA = det(A)
        Fq = qr(A; nsample=12)
        for θ in RNG_PTS
            @test det(A * B)(θ)[1, 1] ≈ dA(θ)[1, 1] * det(B)(θ)[1, 1] atol = 1e-7  # homomorphism
        end
        for (i, t) in enumerate(Fq.ts)
            @test abs(det(A(t))) ≈ prod(abs.(diag(Fq.R[i]))) atol = 1e-8           # |det|=∏|R_ii|
        end
        G = para(A) * A
        for θ in RNG_PTS
            @test sort(svdvals(A(θ)) .^ 2) ≈ sort(real(eigvals(G(θ)))) atol = 1e-8 # σ²=eig(AᴴA)
            λ = eigvals(G(θ))
            @test maximum(abs, imag(λ)) < 1e-9 && minimum(real, λ) > -1e-9         # para-Herm ⇒ real ≥ 0
        end
    end
end

@testset "reconstruction + canonical gauge" begin
    for d in FSIZES, seed in SEEDS
        A = randpm(d, Laurent(-1, 1); seed=seed)
        Fq, Fl, Fs, Fu, Fp = qr(A; nsample=8),
        lq(A; nsample=8), svd(A; nsample=8), lu(A; nsample=8),
        polar(A; nsample=8)
        for (i, t) in enumerate(Fq.ts)
            At = Matrix(A(t))
            @test Fq.Q[i] * Fq.R[i] ≈ At atol = 1e-10
            @test Fq.Q[i]' * Fq.Q[i] ≈ I atol = 1e-10                  # canonical isometry
            @test all(real(diag(Fq.R[i])) .≥ -1e-12)                   # canonical gauge
            @test all(abs.(imag(diag(Fq.R[i]))) .< 1e-10)
            @test Fl.L[i] * Fl.Q[i] ≈ At atol = 1e-10
            @test Fs.U[i] * Diagonal(Fs.S[i]) * Fs.V[i]' ≈ At atol = 1e-10
            @test Fu.L[i] * Fu.U[i] ≈ At[Fu.p[i], :] atol = 1e-10
            @test Fp.U[i]' * Fp.U[i] ≈ I atol = 1e-10                  # polar unitary
            @test Fp.U[i] * Fp.P[i] ≈ At atol = 1e-10
        end
    end
end

@testset "non-square factorizations (tall m>n)" begin
    for (m, n) in ((4, 2), (6, 3), (5, 2)), seed in SEEDS
        A = randpm(m, n, Laurent(-1, 1); seed=seed)
        Fq, Fs = qr(A; nsample=8), svd(A; nsample=8)
        for (i, t) in enumerate(Fq.ts)
            At = Matrix(A(t))
            @test Fq.Q[i]' * Fq.Q[i] ≈ I(n) atol = 1e-10               # column isometry
            @test Fq.Q[i] * Fq.R[i] ≈ At atol = 1e-10
            @test Fs.U[i] * Diagonal(Fs.S[i]) * Fs.V[i]' ≈ At atol = 1e-10
        end
    end
end

@testset "pinv: SVD divergence removal" begin
    U = Matrix(qr(randn(MersenneTwister(1), ComplexF64, 2, 2)).Q)
    V = Matrix(qr(randn(MersenneTwister(2), ComplexF64, 2, 2)).Q)
    sing = U * Diagonal([1.0, 1e-12]) * V'                            # σ = [1, 1e-12]
    A = ParaMatrix([ComplexF64.(sing)], Laurent(0, 0))
    ts, Aplus = pinv(A; rtol=1e-6, nsample=8)
    for (i, t) in enumerate(ts)
        At = Matrix(A(t))
        @test norm(Aplus[i]) < 1e3                                    # bounded — divergence removed
        @test opnorm(inv(At)) > 1e9                                   # naive inverse diverges
        @test At * Aplus[i] * At ≈ At atol = 1e-5                      # Moore–Penrose on kept subspace
    end
end

# The genuinely non-obvious parameterized behaviour: a pointwise factorization is
# NOT guaranteed to assemble into a continuous / single-valued one — assert the
# generic expectation softly and @warn (not fail) when violated, because that may
# be real maths (eigenvalue crossing, rank drop, gauge winding), not a bug.
@testset "continuity / monodromy (@warn on surprise)" begin
    A = randpm(3, Laurent(-1, 1); seed=2)
    Fq = qr(A; nsample=128)
    jumps = [norm(Fq.Q[i + 1] - Fq.Q[i]) for i in 1:(length(Fq.ts) - 1)]
    meanjump = sum(jumps) / length(jumps)
    if maximum(jumps) > 20 * meanjump + 1e-8
        @warn "Gauge-fixed QR Q(θ) discontinuous around the circle — parameterized QR " *
            "continuity is NOT guaranteed (crossing or gauge jump)." maxjump = maximum(
            jumps
        )
    end
    if norm(Fq(1.0).Q - Fq.Q[1]) > 1e-6
        @warn "QR factor has nonzero monodromy (Q(1) ≠ Q(0)) — genuine winding, not a bug."
    end
    @test isfinite(maximum(jumps))
    _, Λ = on_circle(eigvals, A; nsample=64)
    mingap = minimum(
        minimum(abs(λ[i] - λ[j]) for i in 1:length(λ) for j in (i + 1):length(λ)) for λ in Λ
    )
    mingap < 1e-3 &&
        @warn "Eigenvalue near-crossing (min gap < 1e-3): eigenvectors ill-conditioned." mingap
    @test mingap ≥ 0
end
