# tests src/solver/equations.jl — lyapd (Stein), cocycle_exponent, para_solve / \,
# including the no-silent-failure guards.

@testset "lyapd solves the Stein equation (ρ<1)" begin
    for seed in SEEDS
        A = contractive(randpm(3, Laurent(-1, 1); seed=seed); nsample=8)   # ρ(A(θ)) < 1
        C = randpm(3, Laurent(-1, 1); seed=seed + 12)
        Q = para(C) * C
        ts, Xs = lyapd(A, Q; nsample=8)
        for (i, t) in enumerate(ts)
            At, Qt = Matrix(A(t)), Matrix(Q(t))
            @test Xs[i] ≈ At * Xs[i] * At' + Qt atol = 1e-8
        end
    end
    # guard: ρ(A) ≥ 1 has no bounded solution ⇒ error, not garbage
    bigA = ParaMatrix([fill(2.0 + 0im, 1, 1)], Laurent(0, 0))
    @test_throws ArgumentError lyapd(bigA, paraeye(1, ComplexF64, Laurent(0, 0)))

    # cross-check the O(n³) Schur solve against the independent O(n⁶) kron solve
    for seed in SEEDS
        A = contractive(randpm(4, Laurent(-1, 1); seed=seed + 99); nsample=6)   # ρ(A(θ)) < 1
        C = randpm(4, Laurent(-1, 1); seed=seed + 7)
        Q = para(C) * C
        _, Xs = lyapd(A, Q; nsample=6)
        for (i, t) in enumerate(_circle_pts(6))
            At, Qt = Matrix(A(t)), Matrix(Q(t))
            Xkron = reshape((I - kron(conj(At), At)) \ vec(Qt), 4, 4)   # independent algorithm
            @test Xs[i] ≈ (Xkron + Xkron') / 2 atol = 1e-8
        end
    end
end

@testset "cocycle_exponent" begin
    E = ParaMatrix([fill(0.5 + 0im, 1, 1)], Laurent(0, 0))
    @test cocycle_exponent(E, 3, 5) ≈ log(0.5) atol = 1e-12              # constant ⇒ log of it
    Z = ParaMatrix([zeros(ComplexF64, 2, 2)], Laurent(0, 0))
    @test_throws ArgumentError cocycle_exponent(Z, 1, 5)                 # degenerate ⇒ error, not NaN
end

@testset "para_solve / \\ " begin
    for seed in SEEDS
        x0 = randpm(2, Laurent(-2, 2); seed=seed)
        U0 = ParaMatrix(
            [Matrix(qr(randn(MersenneTwister(seed + 5), ComplexF64, 2, 2)).Q)],
            Laurent(0, 0),
        )
        b = U0 * x0
        x, info = para_solve(U0, b; order=2)
        @test info.residual < 1e-8 && info.converged
        @test (U0 \ b)(0.3) ≈ x0(0.3) atol = 1e-7                        # recovers exact Laurent solution
    end
    # \ warns when the Laurent fit cannot converge (rational/too-high-order solution)
    Ai = paraeye(2, ComplexF64, Laurent(0, 0))
    x9 = randpm(2, Laurent(-9, 9); seed=7)                               # order 9 > default order 8
    @test_logs (:warn,) (Ai \ (Ai * x9))
end
