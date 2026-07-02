# tests src/classes/function_basis.jl — the generic ANSATZ class built from an
# explicit list of scalar functions (orthogonal polynomials / special functions /
# any callables). Oracles are independent: closed-form sin/cos/exp values and
# derivatives, agreement with the built-in `Polynomial` ring class, and an
# independent quadrature for the L² norm/integral.

using Zygote

# flatten a Vector{Matrix} gradient + central-difference reference (real entries)
function _fb_flat(g, c0)
    return [
        g[k][a, b] for k in eachindex(c0) for a in axes(c0[k], 1) for b in axes(c0[k], 2)
    ]
end
function _fb_fd(L, c0; h=1e-6)
    return [
        begin
            cp = [copy(x) for x in c0]
            cp[k][a, b] += h
            Lp = L(cp)
            cp[k][a, b] -= 2h
            (Lp - L(cp)) / (2h)
        end for k in eachindex(c0) for a in axes(c0[k], 1) for b in axes(c0[k], 2)
    ]
end

@testset "FunctionBasis: basis / basis_deriv / nbasis" begin
    fb = FunctionBasis([x -> 1.0, sin, cos]; derivs=[x -> 0.0, cos, x -> -sin(x)])
    @test nbasis(fb) == 3
    for p in RNG_PTS
        @test basis(fb, p) ≈ [1.0, sin(p), cos(p)]
        @test basis_deriv(fb, p) ≈ [0.0, cos(p), -sin(p)]
    end
end

@testset "FunctionBasis reproduces Polynomial(2) pointwise (cross-class oracle)" begin
    # a hand-built monomial ansatz must agree with the built-in ring class — proves
    # the class-agnostic core machinery treats a FunctionBasis exactly like any class
    fb = FunctionBasis(
        [x -> 1.0, x -> float(x), x -> float(x)^2];
        derivs=[x -> 0.0, x -> 1.0, x -> 2 * float(x)],
    )
    for d in (1, 2, 3), seed in SEEDS
        C = [randn(MersenneTwister(seed + 137i), ComplexF64, d, d) for i in 1:3]
        A = ParaMatrix(C, fb)
        R = ParaMatrix(C, Polynomial(2))               # same coeffs, built-in class
        for x in (-0.7, 0.0, 0.4, 1.3)
            @test A(x) ≈ R(x)
            @test evaluate_deriv(A, x) ≈ evaluate_deriv(R, x)
        end
    end
end

@testset "FunctionBasis: evaluate_deriv vs finite difference (special functions)" begin
    fb = FunctionBasis([sin, cos, exp]; derivs=[cos, x -> -sin(x), exp])
    A = randpm(2, fb; seed=3)
    for x in (0.13, 0.5, 0.81)
        @test evaluate_deriv(A, x) ≈ (A(x + 1e-6) - A(x - 1e-6)) / 2e-6 atol = 1e-5
    end
end

@testset "FunctionBasis quadrature: L² norm vs independent quadrature" begin
    fb = FunctionBasis([sin, cos, exp]; interval=(0.0, 1.0))
    for d in (1, 2, 3), seed in SEEDS
        A = randpm(d, fb; seed=seed)
        @test norm(A) ≈ l2norm_quad(A) rtol = 5e-3      # Gauss–Legendre Gram vs fine grid
    end
end

@testset "FunctionBasis quadrature: integral vs fine trapezoid" begin
    fb = FunctionBasis([sin, cos, exp]; interval=(0.0, 1.0))
    A = randpm(2, fb; seed=9)
    N = 4001
    xs = range(0, 1; length=N)
    h = 1 / (N - 1)
    ref = h * (sum(A(t) for t in xs) - (A(xs[1]) + A(xs[end])) / 2)   # ∫₀¹ A(x) dx
    @test integral(A) ≈ ref atol = 1e-5
end

@testset "FunctionBasis is an ansatz class: ring ops error clearly" begin
    fb = FunctionBasis([sin, cos]; derivs=[cos, x -> -sin(x)])
    A = randpm(2, fb; seed=1)
    B = randpm(2, fb; seed=2)
    @test_throws ErrorException A * B
    @test_throws ErrorException A ⊗ B
    @test_throws ErrorException A^2
    @test_throws ErrorException one(A)
    @test_throws ErrorException coeff(A, 0)
    # but the purely-structural ops (same class object) still work
    @test (A + B)(0.3) ≈ A(0.3) + B(0.3)
    @test (A ⊕ B)(0.3) ≈ cat(A(0.3), B(0.3); dims=(1, 2))
end

@testset "FunctionBasis: missing optional data errors clearly" begin
    fb = FunctionBasis([sin, cos])                      # no derivs / gram / integral
    A = randpm(2, fb; seed=5)
    @test_throws ArgumentError basis_deriv(fb, 0.3)
    @test_throws ArgumentError evaluate_deriv(A, 0.3)
    @test_throws ArgumentError basis_gram(fb)
    @test_throws ArgumentError norm(A)                  # norm → dot → basis_gram
    @test_throws ArgumentError integral(A)              # → basis_integral
end

@testset "FunctionBasis: constructor validation" begin
    @test_throws ArgumentError FunctionBasis(Function[])                       # empty
    @test_throws ArgumentError FunctionBasis([sin, cos]; derivs=[cos])         # deriv length
    @test_throws ArgumentError FunctionBasis([sin]; gram=zeros(2, 2))          # gram size
    @test_throws ArgumentError FunctionBasis([sin]; integral=[1.0, 2.0])       # integral length
    @test_throws ArgumentError FunctionBasis([sin]; interval=(1.0, 0.0))       # need a < b
    @test_throws ArgumentError FunctionBasis([sin]; interval=(0.0, 1.0), gram=zeros(1, 1))  # both
end

@testset "Zygote ∘ evaluate (FunctionBasis): AD flows through a custom class" begin
    fb = FunctionBasis([sin, cos, exp])
    for x in (0.2, 0.6), seed in SEEDS
        c0 = [randn(MersenneTwister(seed + i), 2, 2) for i in 1:3]
        L(c) = sum(abs2, evaluate(ParaMatrix(c, fb), x))
        @test _fb_flat(Zygote.gradient(L, c0)[1], c0) ≈ _fb_fd(L, c0) atol = 1e-5
    end
end
