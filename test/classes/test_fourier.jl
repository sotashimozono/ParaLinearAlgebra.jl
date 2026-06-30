# tests src/classes/fourier.jl — the real ansatz class (no ring structure).

@testset "Fourier basis values + Gram" begin
    @test basis(Fourier(2), 0.0) ≈ [1.0, 1.0, 1.0, 0.0, 0.0]
    @test basis(Fourier(1), 0.25) ≈ [1.0, cospi(0.5), sinpi(0.5)]
    @test basis(Fourier(0), 0.7) == [1.0]
    @test diag(basis_gram(Fourier(3))) ≈ [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
end

@testset "Fourier is an ansatz class, not a ring" begin
    @test Fourier <: FunctionClass
    @test !(Fourier <: RingClass)
    # an L²-orthonormal-ish weighted real basis: norm via the Gram == quadrature
    for seed in SEEDS
        A = ParaMatrix([randn(MersenneTwister(seed + i), 3, 3) for i in 1:5], Fourier(2))
        @test norm(A) ≈ l2norm_quad(A) rtol = 5e-3
    end
end
