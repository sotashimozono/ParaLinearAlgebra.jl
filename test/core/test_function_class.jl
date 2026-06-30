# tests src/core/function_class.jl — the FunctionClass interface shared by all
# concrete classes (counts, derivative primitive, L² Gram, dim guard).

@testset "nbasis + constructor guards" begin
    @test nbasis(Fourier(0)) == 1
    @test nbasis(Fourier(3)) == 7
    @test nbasis(Laurent(-2, 2)) == 5
    @test nbasis(Laurent(0, 0)) == 1
    @test nbasis(Analytic(3)) == 4
    @test nbasis(Polynomial(0)) == 1
    @test nbasis(ProductClass(Laurent(-1, 1), Polynomial(2))) == 9
    @test_throws ArgumentError Fourier(-1)
    @test_throws ArgumentError Polynomial(-1)
    @test_throws ArgumentError Laurent(2, 1)
end

@testset "basis_deriv vs finite differences (all classes, many points)" begin
    h = 1e-6
    for c in (Fourier(4), Laurent(-3, 3), Polynomial(5), Analytic(3)), p in RNG_PTS
        @test basis_deriv(c, p) ≈ (basis(c, p + h) .- basis(c, p - h)) ./ (2h) atol = 1e-4
    end
    # generic dim guard: single-parameter classes only have axis 1
    @test basis_deriv(Laurent(-1, 1), 0.3, 1) == basis_deriv(Laurent(-1, 1), 0.3)
    @test_throws ArgumentError basis_deriv(Laurent(-1, 1), 0.3, 2)
end

@testset "basis_gram is the L² basis metric" begin
    @test basis_gram(Laurent(-2, 2)) == Matrix{Float64}(I, 5, 5)        # orthonormal
    @test diag(basis_gram(Fourier(2))) ≈ [1.0, 0.5, 0.5, 0.5, 0.5]      # cos/sin ↦ ½
    @test basis_gram(Polynomial(3)) ≈ [1 / (k + l + 1) for k in 0:3, l in 0:3]  # Hilbert
    # each Gram is symmetric positive definite (a genuine inner-product metric)
    for c in (Fourier(3), Laurent(-2, 2), Polynomial(4))
        M = basis_gram(c)
        @test M ≈ M'
        @test isposdef(Symmetric(M))
    end
end

@testset "BigFloat precision (cispi, not 2π·θ)" begin
    @test abs(basis(Laurent(0, 1), big"0.3")[2]) ≈ 1 atol = big"1e-60"
    @test eltype(basis(Laurent(0, 1), big"0.3")) == Complex{BigFloat}
end
