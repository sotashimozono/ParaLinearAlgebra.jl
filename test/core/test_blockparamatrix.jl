# BlockParaMatrix: blocks with DIFFERENT classes and INDEPENDENT parameters.
# Oracles = assembly against the hand-built dense block matrix, and (block-diagonal)
# spectrum/determinant against the union/product of the per-block answers.

@testset "heterogeneous assembly M(a,b) = [A(a); B(b)]" begin
    A = randpm(2, Laurent(-1, 1); seed=1)                                   # depends on a
    B = ParaMatrix([randn(MersenneTwister(i), ComplexF64, 2, 2) for i in 1:5], Fourier(2))  # on b
    M = BlockParaMatrix([[A], [B]])                                         # 2 block-rows
    @test nparams(M) == 2
    @test size(M) == (4, 2)
    for (a, b) in ((0.1, 0.2), (0.5, 0.37), (0.0, 0.99))
        @test M((a, b)) ≈ vcat(A(a), B(b)) atol = 1e-12                     # independent params
    end
    @test_throws ArgumentError M((0.1,))                                    # too few parameters
    @test_throws ArgumentError M((0.1, 0.2, 0.3))                          # too many
    @test_throws DimensionMismatch BlockParaMatrix([[A], [randpm(3, Laurent(-1, 1))]])
end

@testset "block-diagonal: spectrum = ∪ blocks, det = ∏ blocks" begin
    A = randpm(2, Laurent(-1, 1); seed=3)
    B = randpm(3, Polynomial(2); seed=4)                                    # different class & param
    Z12 = zeros(ComplexF64, 2, 3)
    Z21 = zeros(ComplexF64, 3, 2)
    M = BlockParaMatrix([[A, Z12], [Z21, B]])
    @test nparams(M) == 2 && size(M) == (5, 5)
    F = eigen(M; nsample=5)
    @test all(p -> length(p) == 2, F.ts)
    for (p, vals) in zip(F.ts, F.values)
        a, b = p
        @test specmatch(vals, vcat(eigvals(A(a)), eigvals(Matrix(B(b)))))   # eig(M) = eig A ∪ eig B
        @test det(M(p)) ≈ det(A(a)) * det(Matrix(B(b))) atol = 1e-8         # block-diagonal det
    end
end

@testset "mixed Fourier-x / Laurent-θ blocks + constant blocks" begin
    Fx = ParaMatrix([randn(MersenneTwister(i), 2, 2) for i in 1:5], Fourier(2))  # angle x
    Lθ = randpm(2, Laurent(-1, 1); seed=7)                                       # angle θ
    C = ComplexF64[1 0; 0 1]                                                     # constant block
    M = BlockParaMatrix([[Fx, C], [C, Lθ]])
    @test nparams(M) == 2
    for (x, θ) in ((0.2, 0.3), (0.9, 0.6))
        @test M((x, θ)) ≈ [Fx(x) C; C Lθ(θ)] atol = 1e-12
    end
    E = eigen(M; nsample=4)                                                       # callable consistency
    @test specmatch(E((0.2, 0.3)).values, eigvals(M((0.2, 0.3))))
end

@testset "svd reconstruction over the joint grid" begin
    A = randpm(2, Laurent(-1, 1); seed=9)
    B = randpm(2, Laurent(-1, 1); seed=10)
    M = BlockParaMatrix([[A], [B]])
    S = svd(M; nsample=4)
    @test length(S.ts) == 4^2
    for (p, U, s, V) in zip(S.ts, S.U, S.S, S.V)
        @test U * Diagonal(s) * V' ≈ M(p) atol = 1e-10
    end
end
