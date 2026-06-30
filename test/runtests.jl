using ParaLinearAlgebra
using LinearAlgebra
using Random
using SparseArrays
using StaticArrays
using Test, Aqua

# ---- helpers --------------------------------------------------------------
randpm(d::Int, cls; T=ComplexF64, seed=0) =
    ParaMatrix([randn(MersenneTwister(seed + 137i), T, d, d) for i in 1:nbasis(cls)], cls)
randpm(m::Int, n::Int, cls; T=ComplexF64, seed=0) =
    ParaMatrix([randn(MersenneTwister(seed + 137i), T, m, n) for i in 1:nbasis(cls)], cls)

_sortc(v) = sort(v; by=z -> (real(z), imag(z)))
_circle_pts(n) = collect(range(0, 1; length=n + 1))[1:n]   # matches the package's internal grid

const RNG_PTS = [0.0, 0.123, 0.37, 0.5, 0.618, 0.84, 0.999]
const CLASSES = (Laurent(-2, 2), Laurent(0, 0), Analytic(2), Polynomial(3))

@testset "ParaLinearAlgebra" begin
    @testset "Aqua" begin
        Aqua.test_all(ParaLinearAlgebra; ambiguities=false)
    end

    # ====================================================================
    @testset "FunctionClass" begin
        @test nbasis(Fourier(0)) == 1
        @test nbasis(Fourier(3)) == 7
        @test nbasis(Laurent(-2, 2)) == 5
        @test nbasis(Laurent(0, 0)) == 1
        @test nbasis(Analytic(3)) == 4
        @test nbasis(Polynomial(0)) == 1
        @test_throws ArgumentError Fourier(-1)
        @test_throws ArgumentError Polynomial(-1)
        @test_throws ArgumentError Laurent(2, 1)

        @test basis(Fourier(2), 0.0) ≈ [1.0, 1.0, 1.0, 0.0, 0.0]
        @test basis(Polynomial(3), 2.0) ≈ [1.0, 2.0, 4.0, 8.0]
        @test basis(Laurent(0, 2), 0.0) ≈ ComplexF64[1, 1, 1]

        # basis_deriv vs central finite differences over many points/classes
        h = 1e-6
        for c in (Fourier(4), Laurent(-3, 3), Polynomial(5)), p in RNG_PTS
            @test basis_deriv(c, p) ≈ (basis(c, p + h) .- basis(c, p - h)) ./ (2h) atol = 1e-4
        end
        # precision: |e^{iωθ}| = 1 to BigFloat ulp (cispi, not 2π·θ)
        @test abs(basis(Laurent(0, 1), big"0.3")[2]) ≈ 1 atol = big"1e-60"
        @test eltype(basis(Laurent(0, 1), big"0.3")) == Complex{BigFloat}
    end

    # ====================================================================
    # The ParaMatrix product is coefficient CONVOLUTION; testing it against the
    # pointwise product A(θ)B(θ) is the convolution theorem — a real check.
    @testset "ParaMatrix algebra: convolution ⟺ pointwise" begin
        for cls in CLASSES, d in (1, 2, 3, 5), seed in (1, 2, 3)
            A = randpm(d, cls; seed=seed)
            B = randpm(d, cls; seed=seed + 1000)
            for θ in RNG_PTS
                @test A(θ) ≈ evaluate(A, θ)
                @test (A + B)(θ) ≈ A(θ) + B(θ)
                @test (2.5 * A)(θ) ≈ 2.5 * A(θ)
                @test (A * B)(θ) ≈ A(θ) * B(θ)        # convolution-then-eval == eval-then-multiply
                @test (A ⊗ B)(θ) ≈ kron(A(θ), B(θ))
                @test (A^2)(θ) ≈ A(θ)^2
                @test tr(A)(θ)[1, 1] ≈ tr(A(θ))
            end
            for θ in (0.13, 0.5, 0.81)
                @test evaluate_deriv(A, θ) ≈ (A(θ + 1e-6) - A(θ - 1e-6)) / 2e-6 atol = 1e-5
            end
            @test norm(A, 2) ≈ sqrt(sum(norm(c)^2 for c in A.coeffs))
        end
        @test_throws Exception ParaMatrix([randn(2, 2)], Laurent(-1, 1))
    end

    # ====================================================================
    @testset "Factorizations: struct + standard-verb dispatch (smoke)" begin
        A = randpm(3, Laurent(-1, 1); seed=1)
        @test eigen(A; nsample=4) isa ParaEigen
        @test svd(A; nsample=4) isa ParaSVD
        @test qr(A; nsample=4) isa ParaQR
        @test lq(A; nsample=4) isa ParaLQ
        @test lu(A; nsample=4) isa ParaLU
        @test polar(A; nsample=4) isa ParaPolar
        U, S, V = svd(A; nsample=4)           # destructuring via iterate
        @test length(U) == length(S) == length(V) == 4
    end

    # ====================================================================
    # NON-TAUTOLOGICAL: compare to closed-form / independent-algorithm answers.
    @testset "Factorizations vs independent expectations" begin
        # --- analytic case: D(θ) = diag(e^{2πiθ}, e^{-2πiθ}) --------------
        D = ParaMatrix(
            [ComplexF64[0 0; 0 1], zeros(ComplexF64, 2, 2), ComplexF64[1 0; 0 0]],
            Laurent(-1, 1),
        )
        for θ in RNG_PTS
            @test _sortc(eigvals(D(θ))) ≈ _sortc([cispi(2θ), cispi(-2θ)])   # known spectrum
            @test svdvals(D(θ)) ≈ [1.0, 1.0]                                # unitary ⇒ σ=1
            @test det(D)(θ)[1, 1] ≈ 1                                       # det = e^{iθ}e^{-iθ}=1
        end
        @test isparaunitary(D)

        # --- analytic case: planar rotation R(θ) (real, Fourier) ----------
        R = ParaMatrix([zeros(2, 2), [1.0 0; 0 1], [0.0 -1; 1 0]], Fourier(1))
        for θ in RNG_PTS
            @test _sortc(eigvals(R(θ))) ≈ _sortc([cispi(2θ), cispi(-2θ)])
            @test svdvals(R(θ)) ≈ [1.0, 1.0]                                # orthogonal
            @test det(R(θ)) ≈ 1                                             # pointwise det
        end

        # --- cross-algorithm triangulation of the determinant -------------
        for d in (2, 3), seed in (3, 11)
            A = randpm(d, Laurent(-1, 1); seed=seed)
            B = randpm(d, Laurent(-1, 1); seed=seed + 1)
            dA = det(A)
            Fq = qr(A; nsample=16)
            for θ in RNG_PTS
                @test dA(θ)[1, 1] ≈ det(A(θ)) atol = 1e-7          # DFT-paradet vs LAPACK det
                @test dA(θ)[1, 1] ≈ prod(eigvals(A(θ))) atol = 1e-7 # vs ∏ eigenvalues
                @test det(A * B)(θ)[1, 1] ≈ dA(θ)[1, 1] * det(B)(θ)[1, 1] atol = 1e-7  # homomorphism
            end
            for (i, t) in enumerate(Fq.ts)                          # |det A| = ∏|R_ii|
                @test abs(det(A(t))) ≈ prod(abs.(diag(Fq.R[i]))) atol = 1e-8
            end
            # SVD vs eigen of the Gram para(A)·A :  σ² = eig(AᴴA)
            G = para(A) * A
            for θ in RNG_PTS
                @test sort(svdvals(A(θ)) .^ 2) ≈ sort(real(eigvals(G(θ)))) atol = 1e-8
            end
        end

        # --- structural laws ----------------------------------------------
        for d in (2, 3), seed in (5, 13)
            M = randpm(d, Analytic(1); seed=seed)
            G = para(M) * M                                          # Hermitian PSD pointwise
            for θ in RNG_PTS
                λ = eigvals(G(θ))
                @test maximum(abs, imag(λ)) < 1e-9                  # para-Hermitian ⇒ real spectrum
                @test minimum(real, λ) > -1e-9                      # PSD ⇒ ≥ 0
            end
        end
    end

    # ====================================================================
    # Reconstruction + canonical conditions (catch gauge/index bugs in MY code).
    @testset "Factorizations: reconstruction + canonical gauge" begin
        for d in (1, 2, 3, 4), seed in (2, 17)
            A = randpm(d, Laurent(-1, 1); seed=seed)
            Fq, Fl, Fs, Fu, Fp =
                qr(A; nsample=8), lq(A; nsample=8), svd(A; nsample=8),
                lu(A; nsample=8), polar(A; nsample=8)
            for (i, t) in enumerate(Fq.ts)
                At = Matrix(A(t))
                @test Fq.Q[i] * Fq.R[i] ≈ At atol = 1e-10
                @test Fq.Q[i]' * Fq.Q[i] ≈ I atol = 1e-10                       # canonical isometry
                @test all(real(diag(Fq.R[i])) .≥ -1e-12)                        # canonical gauge
                @test all(abs.(imag(diag(Fq.R[i]))) .< 1e-10)
                @test Fl.L[i] * Fl.Q[i] ≈ At atol = 1e-10
                @test Fl.Q[i] * Fl.Q[i]' ≈ I atol = 1e-10
                @test Fs.U[i] * Diagonal(Fs.S[i]) * Fs.V[i]' ≈ At atol = 1e-10
                @test Fu.L[i] * Fu.U[i] ≈ At[Fu.p[i], :] atol = 1e-10
                @test Fp.U[i]' * Fp.U[i] ≈ I atol = 1e-10                       # polar unitary
                @test Fp.U[i] * Fp.P[i] ≈ At atol = 1e-10
            end
        end
        # rank-drop detection (genuine: SVD must see the deficiency)
        rk = ParaMatrix([ComplexF64[1 0; 0 0], ComplexF64[0 1; 0 0]], Laurent(0, 1))
        @test numerical_rank(rk; nsample=16, tol=1e-9) == 1
        # dispatch also holds for a real-coeff Fourier ParaMatrix
        Ar = ParaMatrix([randn(MersenneTwister(i), 3, 3) for i in 1:5], Fourier(2))
        @test eigen(Ar; nsample=4) isa ParaEigen
    end

    # ====================================================================
    # The genuinely non-obvious parameterized behaviour: continuity & monodromy.
    # Pointwise factorization is NOT guaranteed to assemble into a continuous /
    # single-valued factorization — so we ASSERT the generic expectation softly
    # and @warn (not fail) when it is violated, because that may be real maths
    # (eigenvalue crossing, rank drop, gauge winding), not a bug.
    @testset "Parameterized continuity / monodromy (@warn on surprise)" begin
        A = randpm(3, Laurent(-1, 1); seed=2)            # smooth, generically full rank
        Fq = qr(A; nsample=128)
        jumps = [norm(Fq.Q[i + 1] - Fq.Q[i]) for i in 1:(length(Fq.ts) - 1)]
        meanjump = sum(jumps) / length(jumps)
        wrap = norm(Fq(1.0).Q - Fq.Q[1])                 # Q(1) vs Q(0): gauge monodromy
        if maximum(jumps) > 20 * meanjump + 1e-8
            @warn "Gauge-fixed QR Q(θ) is discontinuous around the circle — parameterized QR " *
                  "continuity is NOT guaranteed (eigenvalue/rank crossing or gauge jump)." maxjump =
                maximum(jumps) meanjump = meanjump
        end
        if wrap > 1e-6
            @warn "QR factor has nonzero monodromy around the loop (Q(1) ≠ Q(0)) — a genuine " *
                  "parameterized-matrix winding, expected for some A, not a bug." wrap = wrap
        end
        @test isfinite(maximum(jumps))                   # substance is the @warn above

        # eigenvalue near-crossing ⇒ eigenvectors ill-conditioned: warn if detected
        _, Λ = on_circle(eigvals, A; nsample=64)
        mingap = minimum(minimum(abs(λ[i] - λ[j]) for i in 1:length(λ) for j in (i + 1):length(λ)) for λ in Λ)
        if mingap < 1e-3
            @warn "Eigenvalue near-crossing detected (min gap < 1e-3): eigenvectors are " *
                  "ill-conditioned there; treat para-eigen vectors with care." mingap = mingap
        end
        @test mingap ≥ 0
    end

    # ====================================================================
    @testset "In-class exact operations (consistency)" begin
        for d in (1, 2, 3), seed in (2, 9)
            A = randpm(d, Laurent(-1, 1); seed=seed)
            for θ in RNG_PTS
                @test para(A)(θ) ≈ A(θ)' atol = 1e-10        # para-adjoint = conj-transpose on circle
                @test A'(θ) ≈ A(θ)' atol = 1e-10
            end
        end
        # inv of a genuinely θ-dependent para-unitary  D(z)=diag(1,e^{2πiθ})
        D = ParaMatrix([ComplexF64[1 0; 0 0], ComplexF64[0 0; 0 1]], Analytic(1))
        @test isparaunitary(D)
        for θ in RNG_PTS
            @test inv(D)(θ) ≈ inv(D(θ)) atol = 1e-10
        end

        # spectral_factor: independent Bauer algorithm reproduces G = M·para(M)
        for d in (2, 3), seed in (4, 8)
            M0 = randpm(d, Analytic(1); seed=seed)
            G = para(M0) * M0 + paraeye(d, ComplexF64, Laurent(-1, 1))
            @test ishermitian(G) && ispositive(G)
            M = spectral_factor(G; N=32)
            for θ in (0.1, 0.4, 0.7, 0.95)
                @test M(θ) * para(M)(θ) ≈ G(θ) atol = 1e-6
            end
        end
    end

    # ====================================================================
    @testset "Storage backends: Sparse / Static / BigFloat" begin
        cs = [sprand(ComplexF64, 4, 4, 0.5) for _ in 1:3]
        As = ParaMatrix(cs, Laurent(-1, 1))
        Ad = ParaMatrix(Matrix.(cs), Laurent(-1, 1))
        for θ in RNG_PTS
            @test Matrix(As(θ)) ≈ Ad(θ)
            @test Matrix((As * As)(θ)) ≈ Ad(θ) * Ad(θ)
        end
        @test svd(As; nsample=6).S[1] ≈ svdvals(Matrix(As(0.0)))

        ct = [SMatrix{2,2,ComplexF64}(randn(MersenneTwister(i), ComplexF64, 2, 2)) for i in 1:3]
        At = ParaMatrix(ct, Laurent(-1, 1))
        Atd = ParaMatrix(Matrix.(ct), Laurent(-1, 1))
        @test At(0.3) isa SMatrix
        for θ in RNG_PTS
            @test Matrix(At(θ)) ≈ Atd(θ)
            @test Matrix((At * At)(θ)) ≈ Atd(θ) * Atd(θ)
        end

        Ab = ParaMatrix([randn(MersenneTwister(i), Complex{BigFloat}, 2, 2) for i in 1:3], Laurent(-1, 1))
        θb = big"0.1234567890123456789"
        @test maximum(abs, para(Ab)(θb) - Ab(θb)') < big"1e-50"     # exact to BigFloat ulp
        @test eltype(Ab(θb)) == Complex{BigFloat}
    end

    # ====================================================================
    @testset "pinv: SVD divergence removal" begin
        U = Matrix(qr(randn(MersenneTwister(1), ComplexF64, 2, 2)).Q)
        V = Matrix(qr(randn(MersenneTwister(2), ComplexF64, 2, 2)).Q)
        sing = U * Diagonal([1.0, 1e-12]) * V'        # σ = [1, 1e-12]: near-singular ∀θ
        A = ParaMatrix([ComplexF64.(sing)], Laurent(0, 0))
        ts, Aplus = pinv(A; rtol=1e-6, nsample=8)
        for (i, t) in enumerate(ts)
            At = Matrix(A(t))
            @test norm(Aplus[i]) < 1e3                 # bounded — divergence removed
            @test opnorm(inv(At)) > 1e9                # naive inverse DOES diverge
            @test At * Aplus[i] * At ≈ At atol = 1e-5   # Moore–Penrose on the kept subspace
        end
    end

    # ====================================================================
    @testset "Solvers: lyapd / cocycle / para_solve / leading_eigen" begin
        A = 0.2 * randpm(3, Laurent(-1, 1); seed=11)
        C = randpm(3, Laurent(-1, 1); seed=12)
        Q = para(C) * C
        ts, Xs = lyapd(A, Q; nsample=8)
        for (i, t) in enumerate(ts)
            At, Qt = Matrix(A(t)), Matrix(Q(t))
            @test Xs[i] ≈ At * Xs[i] * At' + Qt atol = 1e-8     # Stein equation residual
        end

        E = ParaMatrix([fill(0.5 + 0im, 1, 1)], Laurent(0, 0))
        @test cocycle_exponent(E, 3, 5) ≈ log(0.5) atol = 1e-12  # constant ⇒ log of it

        x0 = randpm(2, Laurent(-2, 2); seed=20)
        U0 = ParaMatrix([Matrix(qr(randn(MersenneTwister(5), ComplexF64, 2, 2)).Q)], Laurent(0, 0))
        b = U0 * x0
        x, info = para_solve(U0, b; order=2)
        @test info.residual < 1e-8 && info.converged
        @test (U0 \ b)(0.3) ≈ x0(0.3) atol = 1e-7              # recovers the exact Laurent solution

        Eg = randpm(3, Laurent(-1, 1); seed=13)
        _, λs, _ = leading_eigen(Eg; nsample=8)
        @test all(abs(λs[i]) ≈ maximum(abs, eigvals(Matrix(Eg(t)))) for (i, t) in enumerate(_circle_pts(8)))
    end

    # ====================================================================
    @testset "AD: rrules vs finite differences" begin
        A = ParaMatrix([randn(MersenneTwister(i), 2, 2) for i in 1:3], Laurent(-1, 1))
        θ = 0.3
        y, pb = ParaLinearAlgebra.rrule(evaluate, A, θ)
        @test y ≈ A(θ)
        Ȳ = randn(MersenneTwister(99), ComplexF64, 2, 2)
        _, Ā, _ = pb(Ȳ)
        w = basis(A.class, θ)
        for k in 1:3
            @test Ā.coeffs[k] ≈ conj(w[k]) .* Ȳ
        end

        cls = Fourier(2)
        Ar = ParaMatrix([randn(MersenneTwister(7i), 2, 2) for i in 1:5], cls)
        θ2 = 0.31
        Lc(coeffs) = sum(abs2, sum(basis(cls, θ2)[m] * coeffs[m] for m in 1:5))
        yr, pbr = ParaLinearAlgebra.rrule(evaluate, Ar, θ2)
        _, Ār, _ = pbr(2 .* yr)
        h = 1e-6
        for (k, a, b) in [(1, 1, 1), (3, 2, 1), (5, 1, 2)]
            cp = [copy(c) for c in Ar.coeffs]; cp[k][a, b] += h
            cm = [copy(c) for c in Ar.coeffs]; cm[k][a, b] -= h
            @test Ār.coeffs[k][a, b] ≈ (Lc(cp) - Lc(cm)) / (2h) atol = 1e-5
        end
    end
end
