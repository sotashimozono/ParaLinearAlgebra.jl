# tests src/core/chainrules.jl END-TO-END through Zygote — i.e. that the
# ChainRulesCore rrules actually COMPOSE under a real AD backend (gradient of a
# scalar loss vs central finite differences), not just in isolation.
# Mooncake/Enzyme end-to-end + ChainRulesCore→ext are tracked in issue #2.

using Zygote

# flatten a Vector{Matrix} gradient + central-difference reference over real entries
function _flat(g, c0)
    return [
        g[k][a, b] for k in eachindex(c0) for a in axes(c0[k], 1) for b in axes(c0[k], 2)
    ]
end
function _fdgrad(L, c0; h=1e-6)
    return [
        begin
            cp = [copy(x) for x in c0]
            cp[k][a, b] += h
            Lp = L(cp)
            cp[k][a, b] -= 2h
            Lm = L(cp)
            (Lp - Lm) / (2h)
        end for k in eachindex(c0) for a in axes(c0[k], 1) for b in axes(c0[k], 2)
    ]
end

@testset "Zygote ∘ evaluate (Fourier, real)" begin
    cls = Fourier(2)
    for θ in (0.0, 0.3, 0.71), seed in SEEDS
        c0 = [randn(MersenneTwister(seed + i), 2, 2) for i in 1:5]
        L(c) = sum(abs2, evaluate(ParaMatrix(c, cls), θ))
        @test _flat(Zygote.gradient(L, c0)[1], c0) ≈ _fdgrad(L, c0) atol = 1e-5
    end
end

@testset "Zygote ∘ ring ops (Laurent): * , kron, para" begin
    clL = Laurent(-1, 1)
    for seed in SEEDS
        B = ParaMatrix(
            [randn(MersenneTwister(20seed + i), ComplexF64, 2, 2) for i in 1:3], clL
        )
        c0 = [randn(MersenneTwister(3seed + i), 2, 2) for i in 1:3]   # real coeffs ⇒ real ∂
        Lmul(c) = sum(abs2, evaluate(ParaMatrix(c, clL) * B, 0.2))
        Lkron(c) = sum(abs2, evaluate(ParaMatrix(c, clL) ⊗ B, 0.4))
        Lpara(c) = sum(abs2, evaluate(para(ParaMatrix(c, clL)), 0.27))
        @test _flat(Zygote.gradient(Lmul, c0)[1], c0) ≈ _fdgrad(Lmul, c0) atol = 1e-5
        @test _flat(Zygote.gradient(Lkron, c0)[1], c0) ≈ _fdgrad(Lkron, c0) atol = 1e-5
        @test _flat(Zygote.gradient(Lpara, c0)[1], c0) ≈ _fdgrad(Lpara, c0) atol = 1e-5
    end
end

@testset "Zygote ∘ + / scalar* (sum over the orbit)" begin
    cls = Fourier(1)
    for seed in SEEDS
        c0 = [randn(MersenneTwister(7seed + i), 2, 2) for i in 1:3]
        # a loss that exercises scalar* and + (3A = 2A + A) at two points
        function L(c)
            A = ParaMatrix(c, cls)
            B = 2.0 * A + A
            return sum(abs2, evaluate(B, 0.1)) + sum(abs2, evaluate(B, 0.6))
        end
        @test _flat(Zygote.gradient(L, c0)[1], c0) ≈ _fdgrad(L, c0) atol = 1e-5
    end
end

# spectral_factor is mutation-free ⇒ AD flows through the ChainRules `cholesky` rule;
# together with the `inv` rrule (matrix-inverse identity in the para-ring) this makes
# the exact factorizations para_qr / para_lq FULLY differentiable — both the
# canonicalization gauge (R / L) and the para-unitary Q. Validated vs finite diff.
# Real coeffs ⇒ real gradient, matching the real-direction finite difference.
@testset "Zygote ∘ spectral_factor + full para_qr / para_lq (inv rrule)" begin
    for seed in SEEDS
        c0 = [
            Matrix{Float64}(I, 2, 2),                          # A = I + small ⇒ Gram PD
            0.2 .* randn(MersenneTwister(seed), 2, 2),
            0.2 .* randn(MersenneTwister(seed + 1), 2, 2),
        ]
        mk(c) = ParaMatrix(c, Laurent(-1, 1))
        Lsf(c) = sum(abs2, spectral_factor(para(mk(c)) * mk(c); N=8)(0.3))
        @test _flat(Zygote.gradient(Lsf, c0)[1], c0) ≈ _fdgrad(Lsf, c0) atol = 1e-5
        # full para_qr: the gauge R AND the para-unitary Q (Q via the inv rrule)
        function Lqr(c)
            F = para_qr(mk(c); order=30, N=16)
            return sum(abs2, F.Q(0.3)) + sum(abs2, F.R(0.3))
        end
        @test _flat(Zygote.gradient(Lqr, c0)[1], c0) ≈ _fdgrad(Lqr, c0) atol = 1e-5
        Llq(c) = sum(abs2, para_lq(mk(c); order=30, N=16).Q(0.3))   # para_lq Q
        @test _flat(Zygote.gradient(Llq, c0)[1], c0) ≈ _fdgrad(Llq, c0) atol = 1e-5
    end
end

# para_svd/para_eigen are NOT reverse-mode differentiable (non-smooth gauge-fixing) —
# they error clearly under AD; but the gauge-invariant VALUE functions ARE (para_svdvals
# / para_eigvals). z^0 = diag(3,1) keeps σ / bands separated so the value fns are smooth.
@testset "Zygote ∘ para_svdvals / para_eigvals (differentiable); para_svd/para_eigen error" begin
    for seed in SEEDS
        c0 = [
            0.2 .* randn(MersenneTwister(seed), 2, 2),
            Float64[3 0; 0 1] .+ 0.2 .* randn(MersenneTwister(seed + 2), 2, 2),
            0.2 .* randn(MersenneTwister(seed + 1), 2, 2),
        ]
        mk(c) = ParaMatrix(c, Laurent(-1, 1))
        Lsv(c) = sum(abs2, para_svdvals(mk(c); order=8, nsample=128)(0.3))
        @test _flat(Zygote.gradient(Lsv, c0)[1], c0) ≈ _fdgrad(Lsv, c0) atol = 1e-4
        Lev(c) = sum(abs2, para_eigvals(para(mk(c)) * mk(c); order=8, nsample=128)(0.3))
        @test _flat(Zygote.gradient(Lev, c0)[1], c0) ≈ _fdgrad(Lev, c0) atol = 1e-4
        # full para_svd / para_eigen: differentiating raises a clear error
        @test_throws ErrorException Zygote.gradient(
            c -> real(sum(abs2, para_svd(mk(c); order=6).S(0.3))), c0
        )
        @test_throws ErrorException Zygote.gradient(
            c -> real(sum(abs2, para_eigen(para(mk(c)) * mk(c); order=6).D(0.3))), c0
        )
    end
end
