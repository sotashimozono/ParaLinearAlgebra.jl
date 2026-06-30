# Quasiperiodic / fractal STRESS tests. The other suites use well-behaved
# parameterizations; Fibonacci-quasicrystal physics is fractal (Cantor spectra,
# incommensurate periods), so it exercises the machinery in the regime that
# actually matters for that application. Each check has an ANALYTIC oracle that is
# exact (trace map) or holds in the incommensurate limit (Herman bound, ergodicity)
# — and several are evaluated at GOLDEN-MEAN-orbit phases, i.e. irrational-period
# sampling, to show the parameterized objects stay consistent there.

const _FIB = let f = [1, 1]
    for _ in 1:22
        push!(f, f[end] + f[end - 1])
    end
    f
end
const _GOLDEN = 2 / (1 + sqrt(5))     # 1/φ — the most irrational rotation number

# an SL(2) transfer matrix T(θ) = [w0 - 2·w1·cos2πθ   -1 ; 1  0], det ≡ 1, as a
# Laurent ParaMatrix (2w1·cos2πθ = w1·(z + z⁻¹))
function _transfer(w0, w1)
    return ParaMatrix(
        [ComplexF64[-w1 0; 0 0], ComplexF64[w0 -1; 1 0], ComplexF64[-w1 0; 0 0]],
        Laurent(-1, 1),
    )
end

@testset "Fibonacci trace map + Fricke–Vogt invariant (exact, ∀θ incl. golden orbit)" begin
    A = _transfer(0.4, 1.0)                       # two distinct SL(2) tiles
    B = _transfer(0.4, 0.7)
    M = Any[B, A]                                 # Fibonacci words: M_k = M_{k-1}·M_{k-2}
    # capped at k=8: product Laurent COEFFICIENTS grow super-exponentially
    # (≈3→20→280→3.5e4), so a longer chain loses precision on evaluation (intrinsic).
    for k in 3:8
        push!(M, M[k - 1] * M[k - 2])
    end
    θs = vcat(RNG_PTS, [mod(n * _GOLDEN, 1.0) for n in 1:5])   # on/off-grid + irrational
    for θ in θs
        x = [tr(Mk(θ)) / 2 for Mk in M]                       # half-traces (det = 1)
        for k in 4:length(x)
            @test x[k] ≈ 2 * x[k - 1] * x[k - 2] - x[k - 3] atol = 1e-8   # trace map
        end
        I0 = x[3]^2 + x[2]^2 + x[1]^2 - 2x[3] * x[2] * x[1] - 1            # Fricke–Vogt
        for k in 2:(length(x) - 1)
            fv = x[k + 1]^2 + x[k]^2 + x[k - 1]^2 - 2x[k + 1] * x[k] * x[k - 1] - 1
            @test fv ≈ I0 atol = 1e-7                          # conserved along the chain
        end
        @test det(M[end](θ)) ≈ 1 atol = 1e-7                  # products stay in SL(2)
    end
end

@testset "irrational-period sampling is consistent (unique ergodicity / Weyl)" begin
    # Birkhoff average along the golden-mean orbit θ_n = nω equals the circle average:
    # the analytic guarantee that incommensurate sampling is consistent with the
    # parameterized (integral / L²) quantities.
    ω = _GOLDEN
    N = 20000
    for d in (2, 3), seed in SEEDS
        A = randpm(d, Laurent(-2, 2); seed=seed)
        time = sum(tr(A(mod(n * ω, 1.0))) for n in 0:(N - 1)) / N
        @test time ≈ tr(integral(A)) atol = 2e-3              # time-avg == ∫ tr A dθ
        time2 = sum(norm(A(mod(n * ω, 1.0)))^2 for n in 0:(N - 1)) / N
        @test time2 ≈ norm(A)^2 rtol = 5e-3                   # time-avg ‖·‖² == ∫‖A‖²dθ
    end
end

@testset "AAH cocycle Lyapunov: Herman bound, golden-mean convergence, high load" begin
    for λ in (1.5, 2.0, 5.0)
        T = _transfer(0.0, λ)                                  # E = 0 (band centre, in spectrum)
        γs = [cocycle_exponent(T, _FIB[n], _FIB[n + 1]) for n in 8:2:18]   # → 1/φ
        @test all(isfinite, γs)                               # q up to F19 ≈ 4181 stays stable
        @test all(γ -> γ ≥ log(λ) - 1e-4, γs)                 # Herman bound γ(E) ≥ ln λ
        @test abs(γs[end] - log(λ)) < abs(γs[1] - log(λ))     # converges toward ln λ
        @test γs[end] ≈ log(λ) atol = 5e-3                    # localized γ = ln λ on the spectrum
    end
    # Herman bound also holds at a gap energy (γ strictly above ln λ there)
    @test cocycle_exponent(_transfer(1.0, 1.5), _FIB[14], _FIB[15]) ≥ log(1.5) - 1e-4
end
