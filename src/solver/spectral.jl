# solver/spectral.jl — factorization/spectral algorithms on a Laurent ParaMatrix.

"""
    para_gram(A) -> ParaMatrix

The left para-Hermitian Gram `Ã·A` (PSD on the circle).
"""
para_gram(A::ParaMatrix{T,S,<:Laurent}) where {T,S} = para(A) * A

"""
    spectral_factor(G; N=24) -> ParaMatrix

Spectral factorization of a symmetric para-Hermitian PD `G` via Bauer's method
(Cholesky of the block-Toeplitz `T[i,j] = G_{i-j}`). Returns the analytic outer
factor `M` (class `Analytic(hi)`) with `G = M · para(M)` on the circle.
"""
function spectral_factor(G::ParaMatrix{T,S,<:Laurent}; N::Int=24) where {T,S}
    c = G.class
    L = c.hi
    c.lo == -L || error("spectral_factor needs a symmetric window, got $(c)")
    N ≥ L || throw(
        ArgumentError(
            "spectral_factor: Toeplitz size N=$N must be ≥ window half-width L=$L; " *
            "otherwise the block-Toeplitz is incomplete and the factor would be wrong",
        ),
    )
    d = size(G, 1)
    TT = float(T)
    Tb = zeros(TT, (N + 1) * d, (N + 1) * d)
    for i in 0:N, j in 0:N
        m = i - j
        if -L ≤ m ≤ L
            Tb[(i * d + 1):((i + 1) * d), (j * d + 1):((j + 1) * d)] = coeff(G, m)
        end
    end
    Lc = try
        cholesky(Hermitian(Tb)).L
    catch e
        e isa LinearAlgebra.PosDefException || rethrow()
        ispositive(G) || error(
            "spectral_factor: G is not positive on the circle (check ispositive(G))"
        )
        error(
            "spectral_factor: Toeplitz Cholesky failed though G is PSD — G is near-singular or N=$N is too small; try a larger N",
        )
    end
    blk(a, b) = Matrix(Lc[(a * d + 1):((a + 1) * d), (b * d + 1):((b + 1) * d)])
    Mcoeffs = [blk(N, N - k) for k in 0:L]
    return ParaMatrix(Mcoeffs, Analytic(L))
end

"""
    leading_eigen(E; nsample=128) -> (ts, λs, vs)

The leading eigenpair functions `λ(θ), v(θ)` of a parameterized (transfer)
matrix `E`, sampled on the circle — the per-θ Perron environment of an iMPS.
For a single point use `eigen(E(θ))` directly.
"""
function leading_eigen(E::ParaMatrix{T,S,<:Laurent}; nsample::Int=128) where {T,S}
    CT = complex(float(T))
    ts = _circle(nsample)
    λs = Vector{CT}(undef, nsample)
    vs = Vector{Vector{CT}}(undef, nsample)
    for (i, t) in enumerate(ts)
        F = eigen(Matrix(E(t)))
        k = argmax(abs.(F.values))
        λs[i] = F.values[k]
        vs[i] = F.vectors[:, k]
    end
    return ts, λs, vs
end

"""
    para_qr(A; N=24, order=12) -> (; Q, R, residual, isometry)

Parameterized ("polynomial") QR of a tall/square Laurent `ParaMatrix` `A` (`m×n`,
full column rank on the circle): `A = Q·R` with BOTH factors returned **as
ParaMatrices** — `R` the analytic R-factor and `Q` **para-unitary**
(`para(Q)·Q = I`) — not pointwise-sampled. Unlike the sampled `qr`, this is a
genuine parameterized→parameterized factorization.

The R-factor is obtained **exactly** (single parameter) as the spectral factor of
the Gram, `para(A)·A = para(R)·R` (matrix Fejér–Riesz / Bauer), and `Q = A·R⁻¹`.
`residual = max_θ‖A(θ) − Q(θ)R(θ)‖` and `isometry = max_θ‖para(Q)(θ)Q(θ) − I‖`
are reported (both ≈ machine ε for well-conditioned `A`; `Q`'s accuracy is set by
the rational-inverse `order`, which `@warn`s if it does not converge). `R` is the
exact gauge to absorb when canonicalizing a parameterized tensor.
"""
function para_qr(
    A::ParaMatrix{T,S,<:Laurent}; N::Int=24, order::Int=12, rankatol::Real=1e-8
) where {T,S}
    m, n = size(A)
    m ≥ n ||
        throw(DimensionMismatch("para_qr needs a tall/square A (m ≥ n); got $(m)×$(n)"))
    G = para(A) * A                          # n×n para-Hermitian; PD iff A is full column rank
    ispositive(G; tol=rankatol) || error(
        "para_qr: A(θ) is not full column rank on the circle (the Gram para(A)·A drops below " *
        "`rankatol`=$(rankatol)); QR with a para-unitary Q is undefined at a rank drop",
    )
    M = spectral_factor(G; N=N)              # G = M·para(M)
    R = para(M)                              # ⟹ para(R)·R = M·para(M) = G  — the R-factor
    R⁻¹ = inv(R; order=order)                # rational; @warns if the Laurent fit needs more order
    Q = A * R⁻¹
    Id = Matrix{complex(float(T))}(I, n, n)
    grid = range(0, 1; length=(4N + 1))[1:(4N)]
    residual = maximum(norm(Matrix(A(t)) - Matrix(Q(t)) * Matrix(R(t))) for t in grid)
    isometry = maximum(norm(Matrix(Q(t))' * Matrix(Q(t)) - Id) for t in grid)
    return (; Q=Q, R=R, residual=residual, isometry=isometry)
end

"""
    para_lq(A; N=24, order=12, rankatol=1e-8) -> (; L, Q, residual, isometry)

Parameterized LQ of a wide/square Laurent `ParaMatrix` `A` (`m×n`, full **row**
rank on the circle): `A = L·Q` with `L` the analytic L-factor and `Q`
**para-unitary by rows** (`Q·para(Q) = I`), both returned as ParaMatrices. The
mirror of [`para_qr`](@ref): `L` is obtained **exactly** as the spectral factor of
`A·para(A) = L·para(L)`, and `Q = L⁻¹·A`. `residual = max_θ‖A − LQ‖`,
`isometry = max_θ‖Q(θ)para(Q)(θ) − I‖`. `L` is the exact gauge to absorb when
right-canonicalizing a parameterized tensor.
"""
function para_lq(
    A::ParaMatrix{T,S,<:Laurent}; N::Int=24, order::Int=12, rankatol::Real=1e-8
) where {T,S}
    m, n = size(A)
    n ≥ m ||
        throw(DimensionMismatch("para_lq needs a wide/square A (n ≥ m); got $(m)×$(n)"))
    G = A * para(A)                          # m×m para-Hermitian; PD iff A is full row rank
    ispositive(G; tol=rankatol) || error(
        "para_lq: A(θ) is not full row rank on the circle (the Gram A·para(A) drops below " *
        "`rankatol`=$(rankatol)); LQ with a para-unitary Q is undefined at a rank drop",
    )
    L = spectral_factor(G; N=N)              # G = L·para(L) — the L-factor (analytic)
    L⁻¹ = inv(L; order=order)                # rational; @warns if the Laurent fit needs more order
    Q = L⁻¹ * A
    Id = Matrix{complex(float(T))}(I, m, m)
    grid = range(0, 1; length=(4N + 1))[1:(4N)]
    residual = maximum(norm(Matrix(A(t)) - Matrix(L(t)) * Matrix(Q(t))) for t in grid)
    isometry = maximum(norm(Matrix(Q(t)) * Matrix(Q(t))' - Id) for t in grid)
    return (; L=L, Q=Q, residual=residual, isometry=isometry)
end
