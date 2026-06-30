# tests src/classes/product.jl — the multi-parameter ring class (twist-torus /
# many-body-Chern territory): evaluate, 2-D convolution, multi-axis para-adjoint,
# per-axis ∂, and the separable L² Gram.

@testset "ProductClass evaluate + 2-D convolution + para" begin
    pc = ProductClass(Laurent(-1, 1), Laurent(-1, 1))
    @test pc isa RingClass
    @test nbasis(pc) == 9
    for d in (1, 2, 3), seed in SEEDS
        A = ParaMatrix(
            [randn(MersenneTwister(seed + i), ComplexF64, d, d) for i in 1:9], pc
        )
        B = ParaMatrix(
            [randn(MersenneTwister(seed + 20i), ComplexF64, d, d) for i in 1:9], pc
        )
        for ps in ((0.1, 0.7), (0.4, 0.4), (0.83, 0.21))
            @test A(ps) ≈ sum(basis(pc, ps)[k] * A.coeffs[k] for k in 1:9)
            @test (A * B)(ps) ≈ A(ps) * B(ps)            # 2-D convolution theorem
            @test para(A)(ps) ≈ A(ps)'                   # multi-axis para-adjoint
        end
    end
end

@testset "ProductClass per-axis derivative + Gram" begin
    pc = ProductClass(Laurent(-1, 1), Polynomial(2))
    A = ParaMatrix([randn(MersenneTwister(i), ComplexF64, 2, 2) for i in 1:nbasis(pc)], pc)
    h = 1e-6
    ps = (0.3, 0.6)
    d1 = (A((ps[1] + h, ps[2])) - A((ps[1] - h, ps[2]))) / 2h
    d2 = (A((ps[1], ps[2] + h)) - A((ps[1], ps[2] - h))) / 2h
    @test evaluate_deriv(A, ps, 1) ≈ d1 atol = 1e-5
    @test evaluate_deriv(A, ps, 2) ≈ d2 atol = 1e-5
    @test_throws ArgumentError basis_deriv(pc, ps)            # scalar form undefined
    @test_throws ArgumentError basis_deriv(pc, ps, 3)         # axis out of range

    # separable L² Gram vs 2-D quadrature (all-Laurent ⇒ orthonormal ⇒ I)
    AL = ParaMatrix(
        [randn(MersenneTwister(3i), ComplexF64, 2, 2) for i in 1:9],
        ProductClass(Laurent(-1, 1), Laurent(-1, 1)),
    )
    @test norm(AL) ≈ l2norm_quad2(AL) rtol = 5e-3
end

@testset "multi-parameter arity is enforced (no silent scalar mis-eval)" begin
    pc = ProductClass(Fourier(1), Laurent(-1, 1))   # 2 axes ⇒ a 2-tuple is required
    A = ParaMatrix(
        [ComplexF64[i == j ? 1.0 : 0.1 for i in 1:2, j in 1:2] for _ in 1:nbasis(pc)], pc
    )
    # a SCALAR used to silently return a garbage 1-term sum — must now error
    @test_throws ArgumentError basis(pc, 0.2)
    @test_throws ArgumentError A(0.2)
    @test_throws ArgumentError A((0.2,))            # wrong length (too short)
    @test_throws ArgumentError A((0.2, 0.3, 0.4))   # too long
    # correct arity works, and a vector is accepted equivalently to a tuple
    @test A((0.2, 0.3)) ≈ A([0.2, 0.3])
    @test length(basis(pc, (0.2, 0.3))) == nbasis(pc)
    # decomposition now samples the FULL N-D grid (no more silent 1-D garbage):
    # F.ts holds 2-tuples and the callable agrees pointwise with eigen(A(p))
    F = eigen(A; nsample=4)
    @test length(F.ts) == 4^2
    @test all(p -> p isa Tuple && length(p) == 2, F.ts)
    @test F((0.2, 0.3)).values ≈ eigen(A((0.2, 0.3))).values
end
