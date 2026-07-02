# tests ext/ParaLinearAlgebraPolynomialsExt.jl — the Polynomials.jl bridge.
# `using Polynomials` (a test dependency) triggers the package extension, which
# adds `polynomial_basis`, `monomialize`, and the ParaMatrix ⇄ Polynomial
# conversions. `Polynomial` (unqualified) stays the core ring class; the
# Polynomials.jl one is spelled `Polynomials.Polynomial`.

using Polynomials: ChebyshevT
using Polynomials: Polynomials

@testset "polynomial_basis(ChebyshevT): Tₙ(cos πt) = cos(nπt) (independent oracle)" begin
    c = polynomial_basis(ChebyshevT, 6)
    @test nbasis(c) == 7
    # A = the n-th Chebyshev basis polynomial (unit coefficient in slot n)
    for n in 0:6, t in RNG_PTS
        C = [fill(k == n ? 1.0 + 0im : 0.0 + 0im, 1, 1) for k in 0:6]
        @test ParaMatrix(C, c)(cospi(t))[1, 1] ≈ cospi(n * t) atol = 1e-10
    end
end

@testset "polynomial_basis: derivative vs finite difference" begin
    c = polynomial_basis(ChebyshevT, 5)
    A = randpm(2, c; seed=7)
    for x in (-0.8, -0.2, 0.3, 0.77)
        @test evaluate_deriv(A, x) ≈ (A(x + 1e-6) - A(x - 1e-6)) / 2e-6 atol = 1e-5
    end
end

@testset "monomialize: exact change of basis + enables the ring product" begin
    for seed in SEEDS
        A = randpm(2, polynomial_basis(ChebyshevT, 4); seed=seed)
        B = randpm(2, polynomial_basis(ChebyshevT, 3); seed=seed + 100)
        Am, Bm = monomialize(A), monomialize(B)
        @test function_class(Am) == Polynomial(4)
        @test function_class(Bm) == Polynomial(3)
        for x in (-0.9, -0.3, 0.25, 0.8)
            @test Am(x) ≈ A(x) atol = 1e-10                      # values preserved
            @test (Am * Bm)(x) ≈ A(x) * B(x) atol = 1e-9        # ring product == pointwise
        end
    end
end

@testset "exact Gram/integral cross-check the core Polynomial convention" begin
    # the monomial family on [0,1] must reproduce the SAME closed forms the core
    # `Polynomial` class declares: Hilbert Gram 1/(k+l+1) and integral 1/(k+1)
    c = polynomial_basis(Polynomials.Polynomial, 3)
    @test basis_gram(c) ≈ [1 / (k + l + 1) for k in 0:3, l in 0:3] atol = 1e-12
    @test basis_integral(c) ≈ [1 / (k + 1) for k in 0:3] atol = 1e-12
    # ⇒ identical coefficients give the same L² norm on both classes
    A1 = randpm(2, c; seed=4)
    A2 = randpm(2, Polynomial(3); seed=4)
    @test norm(A1) ≈ norm(A2) atol = 1e-10
end

@testset "ParaMatrix ⇄ Polynomials.Polynomial conversions" begin
    q = Polynomials.Polynomial([1.0, -2.0, 3.0])            # 1 - 2x + 3x²
    A = ParaMatrix(q)
    @test function_class(A) == Polynomial(2)
    for x in (-1.0, 0.0, 0.5, 2.0)
        @test A(x)[1, 1] ≈ q(x) atol = 1e-12
    end
    @test Polynomials.coeffs(Polynomials.Polynomial(A)) ≈ Polynomials.coeffs(q)   # round-trip
    # a ChebyshevT linearises to its monomial expansion (2x² − 1)
    T2 = ChebyshevT([0.0, 0.0, 1.0])
    P = ParaMatrix(T2)
    for x in (-0.8, 0.1, 0.9)
        @test P(x)[1, 1] ≈ T2(x) atol = 1e-12
    end
    # the scalar conversion rejects a non-scalar ParaMatrix
    @test_throws ArgumentError Polynomials.Polynomial(randpm(2, Polynomial(2); seed=1))
end

@testset "polynomial_basis: interval override + argument validation" begin
    @test_throws ArgumentError polynomial_basis(ChebyshevT, -1)
    @test_throws ArgumentError polynomial_basis(ChebyshevT, 2; interval=(1.0, 0.0))
    # integral of the constant basis element over [a,b] is (b − a)
    c = polynomial_basis(ChebyshevT, 0; interval=(0.0, 2.0))
    @test basis_integral(c) ≈ [2.0] atol = 1e-12
end
