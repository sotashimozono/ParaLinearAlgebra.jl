# tests src/classes/polynomial.jl — the real-parameter ring class.

@testset "Polynomial basis / deriv / Gram" begin
    @test basis(Polynomial(3), 2.0) ≈ [1.0, 2.0, 4.0, 8.0]
    @test basis(Polynomial(0), 5.0) == [1.0]
    @test basis_deriv(Polynomial(3), 2.0) ≈ [0.0, 1.0, 4.0, 12.0]   # d/dx[1,x,x²,x³]
    @test basis_gram(Polynomial(3)) ≈ [1 / (k + l + 1) for k in 0:3, l in 0:3]
end

@testset "Polynomial ring: convolution ⟺ pointwise product" begin
    for d in (1, 2, 3), seed in SEEDS
        A = randpm(d, Polynomial(2); seed=seed)
        B = randpm(d, Polynomial(2); seed=seed + 500)
        for x in (-0.7, 0.0, 0.4, 1.3)
            @test (A * B)(x) ≈ A(x) * B(x)
            @test (A^0)(x) ≈ I
        end
        @test norm(A) ≈ l2norm_quad(A) rtol = 5e-3       # L² on [0,1] via Hilbert Gram
    end
end
