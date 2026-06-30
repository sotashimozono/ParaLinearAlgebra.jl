# tests the L² inner product `dot` and the parameter integral `integral`
# (src/core/paramatrix.jl + basis_gram/basis_integral). NON-tautological: the
# coefficient formulas are cross-checked against independent quadrature and
# closed forms.

function _dot_quad(A, B; N=2048)
    return sum(dot(Matrix(A(t)), Matrix(B(t))) for t in range(0, 1; length=N + 1)[1:N]) / N
end
_int_quad(A; N=2048) = sum(Matrix(A(t)) for t in range(0, 1; length=N + 1)[1:N]) / N

@testset "dot = ∫⟨A(θ),B(θ)⟩ dθ  (Gram formula vs quadrature)" begin
    for cls in CLASSES, d in (1, 2, 3, 5), seed in SEEDS
        A = randpm(d, cls; seed=seed)
        B = randpm(d, cls; seed=seed + 500)
        # atol floors the near-zero case (Polynomial quadrature is only O(1/N))
        @test dot(A, B) ≈ _dot_quad(A, B) rtol = 5e-3 atol = 2e-3
        @test dot(A, A) ≈ norm(A, 2)^2 rtol = 1e-10          # consistency with norm
        @test real(dot(A, A)) > 0
        @test dot(A, B) ≈ conj(dot(B, A)) rtol = 1e-10       # conjugate symmetry
        # sesquilinearity
        @test dot(3.0 * A, B) ≈ 3.0 * dot(A, B) rtol = 1e-10
        @test dot(A, 2.0 * B) ≈ 2.0 * dot(A, B) rtol = 1e-10
    end
end

@testset "integral = ∫₀¹ A(θ) dθ  (formula vs quadrature + closed forms)" begin
    for cls in CLASSES, d in (1, 2, 3), seed in SEEDS
        A = randpm(d, cls; seed=seed)
        @test integral(A) ≈ _int_quad(A) rtol = 5e-3 atol = 2e-3
    end
    # closed forms
    A = randpm(3, Laurent(-2, 2); seed=1)
    @test integral(A) ≈ coeff(A, 0)                          # Laurent: zero mode
    Af = ParaMatrix([randn(MersenneTwister(i), 2, 2) for i in 1:5], Fourier(2))
    @test integral(Af) ≈ Af.coeffs[1]                        # Fourier: constant term
    Ap = ParaMatrix([randn(MersenneTwister(i), 2, 2) for i in 1:4], Polynomial(3))
    @test integral(Ap) ≈ sum(Ap.coeffs[k + 1] / (k + 1) for k in 0:3)
end

@testset "closed-form: para-unitary U ⇒ ⟨U,U⟩ = ∫ tr(UᴴU) = d" begin
    # D(θ) = diag(e^{2πiθ}, e^{-2πiθ}) is 2×2 unitary ∀θ ⇒ ‖D(θ)‖_F² = 2
    D = ParaMatrix(
        [ComplexF64[0 0; 0 1], zeros(ComplexF64, 2, 2), ComplexF64[1 0; 0 0]],
        Laurent(-1, 1),
    )
    @test dot(D, D) ≈ 2 rtol = 1e-10
    @test integral(D) ≈ zeros(2, 2) atol = 1e-12             # ∫ diag(e^{±iθ}) = 0
end

@testset "ProductClass (multi-parameter) dot + integral" begin
    pc = ProductClass(Laurent(-1, 1), Laurent(-1, 1))
    A = ParaMatrix([randn(MersenneTwister(i), ComplexF64, 2, 2) for i in 1:9], pc)
    B = ParaMatrix([randn(MersenneTwister(7i), ComplexF64, 2, 2) for i in 1:9], pc)
    N = 96
    grid = range(0, 1; length=N + 1)[1:N]
    dq = sum(dot(A((s, t)), B((s, t))) for s in grid, t in grid) / N^2
    iq = sum(A((s, t)) for s in grid, t in grid) / N^2
    @test dot(A, B) ≈ dq rtol = 5e-3
    @test dot(A, A) ≈ norm(A)^2 rtol = 1e-10
    @test integral(A) ≈ iq rtol = 5e-3
end
