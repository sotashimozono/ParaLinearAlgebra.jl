# tests the diagonal surface (diag/diagm) and structured coefficient storage
# (Diagonal/Hermitian/Symmetric) + their closure under the ring operations.

@testset "diag / diagm" begin
    for d in (1, 2, 3, 5), seed in SEEDS
        A = randpm(d, Laurent(-1, 1); seed=seed)
        ds = diag(A)
        @test length(ds) == d
        for θ in (0.0, 0.3, 0.71)
            @test all(ds[i](θ)[1, 1] ≈ A(θ)[i, i] for i in 1:d)
            @test sum(ds)(θ)[1, 1] ≈ tr(A(θ))                  # sum(diag) = tr
            @test diagm(ds)(θ) ≈ Diagonal(diag(A(θ)))          # diagm∘diag = diagonal part
        end
    end
    @test length(diag(randpm(4, 2, Laurent(-1, 1); seed=1))) == 2   # non-square ⇒ min dim
    # diagm guards
    @test_throws ErrorException diagm([randpm(2, Laurent(-1, 1); seed=1)])   # entry not 1×1
    mixed = [
        randpm(2, Laurent(-1, 1); seed=1)[1, 1], randpm(2, Polynomial(1); seed=1)[1, 1]
    ]
    @test_throws ErrorException diagm(mixed)                                  # mixed class
end

@testset "Diagonal coefficients close under +, *, ⊗, evaluate" begin
    for seed in SEEDS
        dc = [Diagonal(randn(MersenneTwister(seed + i), ComplexF64, 4)) for i in 1:3]
        dc2 = [Diagonal(randn(MersenneTwister(seed + 10i), ComplexF64, 4)) for i in 1:3]
        A = ParaMatrix(dc, Laurent(-1, 1))
        B = ParaMatrix(dc2, Laurent(-1, 1))
        Ad = ParaMatrix(Matrix.(dc), Laurent(-1, 1))
        Bd = ParaMatrix(Matrix.(dc2), Laurent(-1, 1))
        for θ in RNG_PTS
            @test A(θ) isa Diagonal                            # structure preserved by evaluate
            @test A(θ) ≈ Ad(θ)
            @test (A * B)(θ) ≈ Ad(θ) * Bd(θ)
            @test (A + B)(θ) ≈ Ad(θ) + Bd(θ)
        end
        @test all(c isa Diagonal for c in (A + B).coeffs)      # closure of the coefficient type
        @test all(c isa Diagonal for c in (A * B).coeffs)
    end
end

@testset "Hermitian / Symmetric coefficient storage works (matches dense)" begin
    for seed in SEEDS
        hc = [Hermitian(randn(MersenneTwister(seed + 3i), ComplexF64, 3, 3)) for i in 1:3]
        Ah = ParaMatrix(hc, Laurent(-1, 1))
        Ahd = ParaMatrix(Matrix.(hc), Laurent(-1, 1))
        sc = [Symmetric(randn(MersenneTwister(seed + 5i), 3, 3)) for i in 1:3]
        As = ParaMatrix(sc, Polynomial(2))
        Asd = ParaMatrix(Matrix.(sc), Polynomial(2))
        for θ in RNG_PTS
            @test Matrix(Ah(θ)) ≈ Ahd(θ)
            @test Matrix(As(θ)) ≈ Asd(θ)
        end
    end
end
