# tests src/core/chainrules.jl — the ChainRulesCore rrules (finite-difference and
# analytic checks). NOTE: differentiation through Zygote/Mooncake/Enzyme is NOT
# yet exercised here — tracked as a dedicated issue (explicit AD impl + tests).

@testset "evaluate rrule scatters with the basis weights" begin
    for seed in SEEDS
        A = ParaMatrix(
            [randn(MersenneTwister(seed + i), 2, 2) for i in 1:3], Laurent(-1, 1)
        )
        θ = 0.3
        y, pb = ParaLinearAlgebra.rrule(evaluate, A, θ)
        @test y ≈ A(θ)
        Ȳ = randn(MersenneTwister(seed + 99), ComplexF64, 2, 2)
        _, Ā, _ = pb(Ȳ)
        w = basis(A.class, θ)
        for k in 1:3
            @test Ā.coeffs[k] ≈ conj(w[k]) .* Ȳ
        end
    end
end

@testset "evaluate rrule vs finite differences (real Fourier loss)" begin
    cls = Fourier(2)
    Ar = ParaMatrix([randn(MersenneTwister(7i), 2, 2) for i in 1:5], cls)
    θ2 = 0.31
    Lc(coeffs) = sum(abs2, sum(basis(cls, θ2)[m] * coeffs[m] for m in 1:5))
    yr, pbr = ParaLinearAlgebra.rrule(evaluate, Ar, θ2)
    _, Ār, _ = pbr(2 .* yr)
    h = 1e-6
    for (k, a, b) in [(1, 1, 1), (3, 2, 1), (5, 1, 2)]
        cp = [copy(c) for c in Ar.coeffs]
        cp[k][a, b] += h
        cm = [copy(c) for c in Ar.coeffs]
        cm[k][a, b] -= h
        @test Ār.coeffs[k][a, b] ≈ (Lc(cp) - Lc(cm)) / (2h) atol = 1e-5
    end
end

@testset "+ rrule gives A,B independent cotangent buffers (Mooncake-safe)" begin
    P = ParaMatrix([randn(MersenneTwister(i), 2, 2) for i in 1:3], Laurent(-1, 1))
    Q = ParaMatrix([randn(MersenneTwister(10i), 2, 2) for i in 1:3], Laurent(-1, 1))
    _, pbplus = ParaLinearAlgebra.rrule(+, P, Q)
    Ȳp = ParaMatrix(
        [randn(MersenneTwister(2i), ComplexF64, 2, 2) for i in 1:3], Laurent(-1, 1)
    )
    _, P̄, Q̄ = pbplus(Ȳp)
    @test all(P̄.coeffs[k] ≈ Ȳp.coeffs[k] for k in 1:3)
    @test P̄.coeffs !== Q̄.coeffs && all(P̄.coeffs[k] ≈ Q̄.coeffs[k] for k in 1:3)
end
