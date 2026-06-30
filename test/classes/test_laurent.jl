# tests src/classes/laurent.jl — the ring class on the circle plus its
# Laurent-specific algebra: para-adjoint, det, inv (para-unitary), predicates.

@testset "Laurent basis + Gram" begin
    @test basis(Laurent(0, 2), 0.0) ≈ ComplexF64[1, 1, 1]
    @test basis(Laurent(-1, 1), 0.25) ≈ ComplexF64[cispi(-0.5), 1, cispi(0.5)]
    @test basis_gram(Laurent(-2, 2)) == Matrix{Float64}(I, 5, 5)
    @test Analytic(3) == Laurent(0, 3)
end

@testset "para-adjoint = conj-transpose on the circle" begin
    for d in (1, 2, 3, 5), seed in SEEDS
        A = randpm(d, Laurent(-2, 2); seed=seed)
        for θ in RNG_PTS
            @test para(A)(θ) ≈ A(θ)' atol = 1e-10
            @test A'(θ) ≈ A(θ)' atol = 1e-10            # adjoint dispatches to para
        end
        @test para(para(A)) ≈ A                         # involution
    end
end

@testset "det(A)(θ) == det(A(θ))  (independent algorithms)" begin
    for d in (1, 2, 3), seed in SEEDS
        A = randpm(d, Laurent(-1, 1); seed=seed)
        dA = det(A)
        for θ in RNG_PTS
            @test dA(θ)[1, 1] ≈ det(A(θ)) atol = 1e-7        # DFT-paradet vs LAPACK
            @test dA(θ)[1, 1] ≈ prod(eigvals(A(θ))) atol = 1e-7   # vs ∏ eigenvalues
        end
    end
end

@testset "predicates + para-unitary inverse" begin
    for seed in SEEDS
        A = randpm(3, Laurent(-1, 1); seed=seed)
        H = para(A) * A                                  # para-Hermitian PSD
        @test isparahermitian(H) && ishermitian(H) && ispositive(H)
        @test !isparahermitian(A)                        # generic A is not
        @test !isparaunitary(H)                          # distinct predicate
    end
    # genuinely θ-dependent para-unitary  D(z) = diag(1, e^{2πiθ})
    D = ParaMatrix([ComplexF64[1 0; 0 0], ComplexF64[0 0; 0 1]], Analytic(1))
    @test isparaunitary(D)
    for θ in RNG_PTS
        @test inv(D)(θ) ≈ inv(D(θ)) atol = 1e-10
    end
    @test_throws ErrorException inv(randpm(2, Laurent(-1, 1); seed=3))  # not para-unitary
end

@testset "parahermitianpart + para-Hermitian closure" begin
    for d in (1, 2, 3), seed in SEEDS
        A = randpm(d, Laurent(-2, 2); seed=seed)            # symmetric window
        H = parahermitianpart(A)
        @test isparahermitian(H)                            # projector lands in the set
        for θ in RNG_PTS
            @test H(θ) ≈ (A(θ) + A(θ)') / 2 atol = 1e-10    # = pointwise Hermitian part
        end
        # closure: para-Hermitian is closed under + and real scaling
        H2 = parahermitianpart(randpm(d, Laurent(-2, 2); seed=seed + 31))
        @test isparahermitian(H + H2)
        @test isparahermitian(3.0 * H)
        @test isparahermitian(para(A) * A)                  # the Gram is para-Hermitian
    end
    # needs a symmetric window
    @test_throws ArgumentError parahermitianpart(randpm(2, Analytic(2); seed=1))
end
