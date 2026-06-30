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
