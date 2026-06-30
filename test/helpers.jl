# test/helpers.jl — shared fixtures, included once by runtests.jl before the
# directory-walked test files. The test files mirror src/: test/core ↔ src/core,
# test/classes ↔ src/classes, test/solver ↔ src/solver, test/utils ↔ src/utils.

# random square / rectangular ParaMatrix (distinct seed per coefficient block)
function randpm(d::Int, cls; T=ComplexF64, seed=0)
    return ParaMatrix(
        [randn(MersenneTwister(seed + 137i), T, d, d) for i in 1:nbasis(cls)], cls
    )
end
function randpm(m::Int, n::Int, cls; T=ComplexF64, seed=0)
    return ParaMatrix(
        [randn(MersenneTwister(seed + 137i), T, m, n) for i in 1:nbasis(cls)], cls
    )
end

_sortc(v) = sort(v; by=z -> (real(z), imag(z)))
_circle_pts(n) = collect(range(0, 1; length=n + 1))[1:n]   # = the package's internal grid

# order-independent spectrum match: greedily pair each computed eigenvalue with the
# nearest expected one. Robust to LAPACK ulp-noise that flips sort tie-breaks (e.g.
# a conjugate pair a±bi whose real parts are equal by construction).
function specmatch(computed, expected; atol=1e-8)
    length(computed) == length(expected) || return false
    pool = collect(ComplexF64, expected)
    for x in computed
        i = argmin(abs.(pool .- x))
        abs(pool[i] - x) ≤ atol || return false
        deleteat!(pool, i)
    end
    return true
end

# the L² function norm by independent quadrature (cross-checks `norm(::ParaMatrix)`)
function l2norm_quad(A; N=2048)
    pts = range(0, 1; length=N + 1)[1:N]
    return sqrt(sum(norm(A(t))^2 for t in pts) / N)
end
function l2norm_quad2(A; N=128)                       # 2-D (ProductClass) version
    pts = range(0, 1; length=N + 1)[1:N]
    return sqrt(sum(norm(A((s, t)))^2 for s in pts, t in pts) / N^2)
end

# scale a Laurent ParaMatrix so that max_θ ‖A(θ)‖₂ = bound < 1 on the nsample grid,
# guaranteeing ρ(A(θ)) < 1 (needed for the discrete-Lyapunov / Stein solver).
function contractive(A; bound=0.7, nsample=8)
    m = maximum(opnorm(Matrix(A(t))) for t in _circle_pts(nsample))
    return (bound / m) * A
end

const RNG_PTS = [0.0, 0.123, 0.37, 0.5, 0.618, 0.84, 0.999]   # on/off-grid + edges
const CLASSES = (Laurent(-2, 2), Laurent(0, 0), Analytic(2), Polynomial(3))

# stress configuration: run randomized tests over several seeds and larger sizes
const STRESS = 5
const SEEDS = (11, 23, 47, 71, 97)            # 5 independent trials
const SIZES = (1, 2, 3, 5, 8)                 # core-algebra matrix sizes
const FSIZES = (1, 2, 3, 4, 6)               # factorization matrix sizes
