# tests src/core/paramatrix.jl — the central type: callable evaluation, the ring
# algebra (convolution ⟺ pointwise product), ∂_p, the L² norm, the RingClass /
# ansatz contract, identity/^0, constructor invariants, and storage backends.
# Randomized over SEEDS (5 trials) and larger SIZES for stress.

@testset "callable + ring algebra (convolution ⟺ pointwise)" begin
    for cls in CLASSES, d in SIZES, seed in SEEDS
        A = randpm(d, cls; seed=seed)
        B = randpm(d, cls; seed=seed + 1000)
        for θ in RNG_PTS
            @test A(θ) ≈ evaluate(A, θ)
            @test (A + B)(θ) ≈ A(θ) + B(θ)
            @test (A - B)(θ) ≈ A(θ) - B(θ)
            @test (2.5 * A)(θ) ≈ 2.5 * A(θ)
            @test (A * B)(θ) ≈ A(θ) * B(θ)            # the convolution theorem
            @test (A ⊗ B)(θ) ≈ kron(A(θ), B(θ))
            @test (A^2)(θ) ≈ A(θ)^2
            @test tr(A)(θ)[1, 1] ≈ tr(A(θ))
        end
        for θ in (0.13, 0.5, 0.81)
            @test evaluate_deriv(A, θ) ≈ (A(θ + 1e-6) - A(θ - 1e-6)) / 2e-6 atol = 1e-5
        end
        @test norm(A, 2) ≈ l2norm_quad(A) rtol = 5e-3   # Gram formula vs quadrature
    end
end

@testset "constructor invariants" begin
    @test_throws Exception ParaMatrix([randn(2, 2)], Laurent(-1, 1))           # wrong count
    @test_throws Exception ParaMatrix(
        [randn(2, 2), randn(3, 3), randn(2, 2)], Laurent(-1, 1)
    )  # mixed sizes
end

@testset "RingClass vs ansatz contract" begin
    @test Laurent <: RingClass && Polynomial <: RingClass && ProductClass <: RingClass
    @test !(Fourier <: RingClass) && Fourier <: FunctionClass
    F = ParaMatrix([randn(MersenneTwister(i), 2, 2) for i in 1:5], Fourier(2))  # ansatz, non-ring
    @test_throws ErrorException F * F
    @test_throws ErrorException F ⊗ F
    @test_throws ErrorException F^2
    @test_throws ErrorException coeff(F, 0)
    @test_throws ErrorException one(F)
    # evaluate / ∂_p still work for the ansatz class
    @test F(0.3) ≈ evaluate(F, 0.3)
    @test evaluate_deriv(F, 0.3) ≈ (F(0.3 + 1e-6) - F(0.3 - 1e-6)) / 2e-6 atol = 1e-5
end

@testset "one / ^0 over ring classes" begin
    for cls in (Laurent(-1, 1), Polynomial(2), Analytic(2)), seed in SEEDS
        A = randpm(3, cls; seed=seed)
        @test (A^0)(0.37) ≈ I
        @test one(A)(0.37) ≈ I
    end
    @test_throws ErrorException one(randpm(2, Laurent(1, 2); seed=1))   # no zero power
end

@testset "accessors" begin
    c0, c1, c2 = [randn(ComplexF64, 3, 3) for _ in 1:3]
    A = ParaMatrix([c0, c1, c2], Laurent(-1, 1))
    @test coeff(A, -1) === c0 && coeff(A, 0) === c1 && coeff(A, 1) === c2   # index arithmetic
    @test nterms(A) == 3
    @test coefficients(A) === A.coeffs
    @test function_class(A) === A.class
    @test size(A) == (3, 3) && eltype(A) == ComplexF64
end

@testset "storage backends: Sparse / Static / BigFloat" begin
    for seed in SEEDS
        cs = [sprand(MersenneTwister(seed + i), ComplexF64, 5, 5, 0.5) for i in 1:3]
        As = ParaMatrix(cs, Laurent(-1, 1))
        Ad = ParaMatrix(Matrix.(cs), Laurent(-1, 1))
        for θ in RNG_PTS
            @test Matrix(As(θ)) ≈ Ad(θ)
            @test Matrix((As * As)(θ)) ≈ Ad(θ) * Ad(θ)
        end

        ct = [
            SMatrix{3,3,ComplexF64}(randn(MersenneTwister(seed + 7i), ComplexF64, 3, 3)) for
            i in 1:3
        ]
        At = ParaMatrix(ct, Laurent(-1, 1))
        Atd = ParaMatrix(Matrix.(ct), Laurent(-1, 1))
        @test At(0.3) isa SMatrix
        for θ in RNG_PTS
            @test Matrix(At(θ)) ≈ Atd(θ)
            @test Matrix((At * At)(θ)) ≈ Atd(θ) * Atd(θ)
        end
    end
    # BigFloat: para-adjoint = conj-transpose to BigFloat ulp
    Ab = ParaMatrix(
        [randn(MersenneTwister(i), Complex{BigFloat}, 2, 2) for i in 1:3], Laurent(-1, 1)
    )
    θb = big"0.1234567890123456789"
    @test maximum(abs, para(Ab)(θb) - Ab(θb)') < big"1e-50"
    @test eltype(Ab(θb)) == Complex{BigFloat}
end
