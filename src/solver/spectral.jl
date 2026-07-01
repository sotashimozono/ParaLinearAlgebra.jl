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

**Differentiable**: the block-Toeplitz is built mutation-free, so reverse-mode AD
(e.g. Zygote) flows through the `cholesky` rule. Together with the `inv` rrule
(matrix-inverse identity in the para-ring) this makes `para_qr`/`para_lq` **fully
differentiable** in the input's coefficients — both the canonicalization gauge
(`R`/`L`) and the para-unitary `Q`.
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
    # block-Toeplitz `T[i,j] = G_{i-j}`, built mutation-free (hcat/vcat of blocks) so
    # it is AD-transparent — `spectral_factor` (and `para_qr`/`para_lq` built on it)
    # then differentiate through the ChainRules `cholesky` rule.
    Z = zeros(TT, d, d)
    blkin(m) = (-L ≤ m ≤ L) ? Matrix{TT}(coeff(G, m)) : Z
    Tb = reduce(vcat, [reduce(hcat, [blkin(i - j) for j in 0:N]) for i in 0:N])
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
    # diagnostics only ⇒ keep them out of the AD graph (a plain `range` has no Zygote
    # adjoint, and residual/isometry are not differentiable outputs).
    residual, isometry = ChainRulesCore.@ignore_derivatives begin
        Id = Matrix{complex(float(T))}(I, n, n)
        grid = range(0, 1; length=(4N + 1))[1:(4N)]
        (
            maximum(norm(Matrix(A(t)) - Matrix(Q(t)) * Matrix(R(t))) for t in grid),
            maximum(norm(Matrix(Q(t))' * Matrix(Q(t)) - Id) for t in grid),
        )
    end
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
    residual, isometry = ChainRulesCore.@ignore_derivatives begin   # diagnostics: out of the AD graph
        Id = Matrix{complex(float(T))}(I, m, m)
        grid = range(0, 1; length=(4N + 1))[1:(4N)]
        (
            maximum(norm(Matrix(A(t)) - Matrix(L(t)) * Matrix(Q(t))) for t in grid),
            maximum(norm(Matrix(Q(t)) * Matrix(Q(t))' - Id) for t in grid),
        )
    end
    return (; L=L, Q=Q, residual=residual, isometry=isometry)
end

# smallest gap between adjacent (sorted) band values over the whole grid — a
# crossing/near-degeneracy diagnostic for the analytic factorizations
function _mingap(vals::AbstractVector)
    g = Inf
    for v in vals
        s = sort(real.(v))
        for k in 1:(length(s) - 1)
            g = min(g, s[k + 1] - s[k])
        end
    end
    return g
end

# greedy column matching: permutation aligning the columns of `cur` with `prev`
# by largest singular-vector overlap (branch tracking through near-degeneracies)
function _svdmatch(prev::AbstractMatrix, cur::AbstractMatrix)
    r = size(cur, 2)
    ov = abs.(prev' * cur)
    perm = zeros(Int, r)
    used = falses(r)
    for k in 1:r
        best, bestv = 0, -1.0
        for j in 1:r
            if !used[j] && ov[k, j] > bestv
                bestv, best = ov[k, j], j
            end
        end
        perm[k] = best
        used[best] = true
    end
    return perm
end

"""
    para_svd(A; order=12, nsample=0, tol=0, maxorder=64, gaptol=1e-4)
        -> (; U, S, V, residual, winding, mingap, order)

Parameterized ("analytic") SVD of a Laurent `ParaMatrix` `A`: factors returned **as
ParaMatrices**, `A(θ) ≈ U(θ)·S(θ)·V(θ)'`, with `U`/`V` column-orthonormal at each
`θ` and `S` diagonal with the **singular-value functions** on its diagonal. Unlike
the sampled `svd`, this is a genuine parameterized→parameterized factorization.

Unlike `para_qr`/`para_lq` it is **necessarily approximate**: the singular
values/vectors of `A(θ)` are analytic but in general *not* finite Laurent
polynomials (infinite series; branch points at crossings). The method samples the
SVD on a grid, **gauge-fixes** it for continuity (branch-tracks the singular
values, parallel-transports the singular vectors with a common `U,V` phase so the
product is invariant), then **DFT-fits** each factor to `Laurent(order)`.

`S` is robust — the singular-value functions are gauge-invariant and periodic, so
they fit cleanly. `U,V` are good **iff** the singular values stay separated and the
per-band Berry phase `winding` vanishes; otherwise the factor is not a band-limited
Laurent and `residual = max_θ‖A − U S V'‖` is large (a `@warn` fires).

Refinement knobs/diagnostics: with `tol > 0` the `order` is grown (re-fitting the
gauge-fixed samples) until `residual ≤ tol` or `order = maxorder`, and the order
actually used is returned. `mingap` is the smallest gap between adjacent singular
values over the grid; if `mingap < gaptol` a crossing/near-degeneracy is reported
(`@warn`) — there the branch tracking and the analytic factor are unreliable.
`nsample` defaults to `max(64, 16·order)`.
"""
function para_svd(
    A::ParaMatrix{T,Sm,<:Laurent};
    order::Int=12,
    nsample::Int=0,
    tol::Real=0,
    maxorder::Int=64,
    gaptol::Real=1e-4,
) where {T,Sm}
    m, n = size(A)
    r = min(m, n)
    N = nsample > 0 ? nsample : max(64, 16 * order)
    CT = complex(float(T))
    grid = range(0, 1; length=(N + 1))[1:N]
    Us = Vector{Matrix{CT}}(undef, N)
    Ss = Vector{Vector{Float64}}(undef, N)
    Vs = Vector{Matrix{CT}}(undef, N)
    for (i, t) in enumerate(grid)
        F = svd(Matrix(A(t)))
        Us[i] = F.U[:, 1:r]
        Ss[i] = F.S[1:r]
        Vs[i] = F.V[:, 1:r]
    end
    # branch-track ordering + parallel-transport phase gauge (same phase on U and V)
    for i in 2:N
        p = _svdmatch(Us[i - 1], Us[i])
        Us[i] = Us[i][:, p]
        Vs[i] = Vs[i][:, p]
        Ss[i] = Ss[i][p]
        for k in 1:r
            ph = dot(Us[i - 1][:, k], Us[i][:, k])
            ph = iszero(ph) ? one(CT) : ph / abs(ph)
            @views Us[i][:, k] .*= conj(ph)
            @views Vs[i][:, k] .*= conj(ph)
        end
    end
    winding = [angle(dot(Us[N][:, k], Us[1][:, k])) for k in 1:r]   # per-band wrap holonomy
    mingap = _mingap(Ss)
    mingap < gaptol && @warn(
        "para_svd: singular values approach within mingap=$(mingap) < gaptol — a " *
            "crossing/near-degeneracy; branch tracking and the analytic U,V are unreliable there.",
        maxlog = 2,
    )
    Sseq = [Matrix{CT}(Diagonal(Ss[i])) for i in 1:N]
    fit(seq, ord) = ParaMatrix(
        [
            sum(seq[j + 1] * cispi(-2 * k * (j / N)) for j in 0:(N - 1)) / N for
            k in (-ord):ord
        ],
        Laurent(-ord, ord),
    )
    g2 = range(0, 1; length=(2N + 1))[1:(2N)]
    # adaptive order: grow until the reconstruction residual meets `tol` (tol≤0 ⇒ fixed `order`)
    ord = min(order, maxorder)
    local U, V, Sd, residual
    while true
        U, V, Sd = fit(Us, ord), fit(Vs, ord), fit(Sseq, ord)
        residual = maximum(
            norm(Matrix(A(t)) - Matrix(U(t)) * Matrix(Sd(t)) * Matrix(V(t))') for t in g2
        )
        (tol ≤ 0 || residual ≤ tol || ord ≥ maxorder || 2 * (ord + 4) ≥ N) && break
        ord += 4
    end
    residual > 1e-6 && @warn(
        "para_svd: residual $(residual) — factors may not be band-limited Laurent " *
            "(singular-value crossing or nonzero per-band Berry phase $(round.(winding; digits=3))); " *
            "raise `order`/`nsample`/`tol`, or expect only the singular values to be accurate.",
        maxlog = 3,
    )
    return (; U=U, S=Sd, V=V, residual=residual, winding=winding, mingap=mingap, order=ord)
end

"""
    para_eigen(H; order=12, nsample=0, tol=0, maxorder=64, gaptol=1e-4)
        -> (; U, D, residual, winding, mingap, order)

Parameterized ("analytic") eigendecomposition (PEVD) of a **para-Hermitian**
Laurent `ParaMatrix` `H` (`H(θ)` Hermitian ∀θ): `H(θ) ≈ U(θ)·D(θ)·U(θ)'` with `U`
para-unitary and `D` **real** diagonal carrying the **eigenvalue functions**
(bands), returned AS ParaMatrices. The eigen-analogue of [`para_svd`](@ref); the
sampled counterpart is `eigen`.

Same approximate nature and honest limits as `para_svd`: eigenvalues/eigenvectors
are analytic but generally **not** finite Laurent (infinite series; branch points
at crossings). `D` (the eigenvalue functions) is gauge-invariant and periodic ⇒
**always recovered**; `U` is a clean Laurent factor only when the bands stay
separated and the per-band Berry/Zak phase `winding ≈ 0`, otherwise `residual` is
large and a `@warn` fires. Unlike `para_svd` the eigenvalues are real and may be
negative, and the same `U` appears on both sides (so reconstruction is invariant
to each eigenvector's phase).

!!! note "Known limitations (scientific scope)"
    Single parameter (Laurent) only; assumes `H` para-Hermitian (else the Hermitian
    part at each θ is used, with a warning). See the `TODO`s in the source for the
    eigenvalue-crossing / degeneracy / ordering-convention cases not yet handled.
"""
function para_eigen(
    H::ParaMatrix{T,S,<:Laurent};
    order::Int=12,
    nsample::Int=0,
    tol::Real=0,
    maxorder::Int=64,
    gaptol::Real=1e-4,
) where {T,S}
    n = size(H, 1)
    n == size(H, 2) ||
        throw(DimensionMismatch("para_eigen needs a square H; got $(size(H))"))
    isparahermitian(H) || @warn(
        "para_eigen: H is not para-Hermitian (H(θ) not Hermitian on the circle); " *
            "decomposing the Hermitian part at each θ instead.",
        maxlog = 1,
    )
    N = nsample > 0 ? nsample : max(64, 16 * order)
    CT = complex(float(T))
    grid = range(0, 1; length=(N + 1))[1:N]
    Us = Vector{Matrix{CT}}(undef, N)
    Ds = Vector{Vector{Float64}}(undef, N)
    for (i, t) in enumerate(grid)
        F = eigen(Hermitian(Matrix(H(t))))   # real, ascending eigenvalues + orthonormal vectors
        Us[i] = F.vectors
        Ds[i] = F.values
    end
    # Branch-track the bands by eigenvector overlap, then parallel-transport each
    # eigenvector's phase for continuity (the eigenvalue-phase cancels in U D U').
    # TODO: at band CROSSINGS / degeneracies the eigenvectors rotate arbitrarily fast
    #   (Wedin/Davis–Kahan: gap→0) and overlap-matching can mis-track; a proper
    #   analytic continuation through crossings is not done. TODO: expose an
    #   ordering convention (analytic branches vs ascending/spectrally-majorised) —
    #   currently analytic-branch via tracking. TODO: degenerate bands leave the
    #   eigenvector gauge ambiguous within the eigenspace (only the subspace is
    #   defined); per-band fitting is then unreliable. TODO: multi-parameter
    #   (ProductClass) PEVD — none here (the 1-D elementary-factor structure that
    #   makes this work has no general multivariate analogue).
    for i in 2:N
        p = _svdmatch(Us[i - 1], Us[i])
        Us[i] = Us[i][:, p]
        Ds[i] = Ds[i][p]
        for k in 1:n
            ph = dot(Us[i - 1][:, k], Us[i][:, k])
            ph = iszero(ph) ? one(CT) : ph / abs(ph)
            @views Us[i][:, k] .*= conj(ph)
        end
    end
    winding = [angle(dot(Us[N][:, k], Us[1][:, k])) for k in 1:n]   # per-band wrap holonomy (Zak)
    mingap = _mingap(Ds)
    mingap < gaptol && @warn(
        "para_eigen: bands approach within mingap=$(mingap) < gaptol — a crossing/" *
            "near-degeneracy; branch tracking and the analytic U are unreliable there.",
        maxlog = 2,
    )
    Dseq = [Matrix{CT}(Diagonal(Ds[i])) for i in 1:N]
    fit(seq, ord) = ParaMatrix(
        [
            sum(seq[j + 1] * cispi(-2 * k * (j / N)) for j in 0:(N - 1)) / N for
            k in (-ord):ord
        ],
        Laurent(-ord, ord),
    )
    g2 = range(0, 1; length=(2N + 1))[1:(2N)]
    ord = min(order, maxorder)
    local U, D, residual
    while true
        U, D = fit(Us, ord), fit(Dseq, ord)
        residual = maximum(
            norm(Matrix(H(t)) - Matrix(U(t)) * Matrix(D(t)) * Matrix(U(t))') for t in g2
        )
        (tol ≤ 0 || residual ≤ tol || ord ≥ maxorder || 2 * (ord + 4) ≥ N) && break
        ord += 4
    end
    residual > 1e-6 && @warn(
        "para_eigen: residual $(residual) — eigenvectors may not be band-limited Laurent " *
            "(band crossing or nonzero per-band Berry phase $(round.(winding; digits=3))); " *
            "raise `order`/`nsample`/`tol`, or expect only the eigenvalue functions `D` to be accurate.",
        maxlog = 3,
    )
    return (; U=U, D=D, residual=residual, winding=winding, mingap=mingap, order=ord)
end

# ---------- differentiable value functions (para_svdvals / para_eigvals) ----------
# The singular values / eigenvalues are GAUGE-INVARIANT, so — unlike the full
# `para_svd`/`para_eigen` (whose vector gauge-fixing is non-smooth and NOT AD-able) —
# their FUNCTIONS fit and differentiate cleanly (per-θ svdvals/eigvals have ChainRules
# rrules; the DFT-fit is linear). Diagonal comprehension (no `Diagonal`/`range`) keeps
# it AD-transparent. Still approximate: band crossings give kinks ⇒ raise `order`.
function _valfit(vals, r, order, N, ::Type{CT}) where {CT}
    return [
        [
            (
                if a == b
                    sum(vals[j + 1][a] * cispi(-2 * k * (j / N)) for j in 0:(N - 1)) / N
                else
                    zero(CT)
                end
            ) for a in 1:r, b in 1:r
        ] for k in (-order):order
    ]
end

"""
    para_svdvals(A; order=12, nsample=0) -> ParaMatrix

The singular-value **functions** of a Laurent `ParaMatrix` `A` as a diagonal
ParaMatrix (descending `σₖ(θ)` on the diagonal), via per-θ `svdvals` + DFT-fit.
Unlike `para_svd` this is **differentiable** (singular values are gauge-invariant,
no vector gauge-fixing), so it composes under reverse-mode AD. Approximate — fits to
`Laurent(order)`; band crossings give kinks, so raise `order`. `nsample` defaults to
`max(64, 16·order)`.
"""
function para_svdvals(
    A::ParaMatrix{T,S,<:Laurent}; order::Int=12, nsample::Int=0
) where {T,S}
    r = minimum(size(A))
    N = nsample > 0 ? nsample : max(64, 16 * order)
    grid = [j / N for j in 0:(N - 1)]
    sv = [svdvals(Matrix(A(t))) for t in grid]           # per-θ, sorted descending
    return ParaMatrix(_valfit(sv, r, order, N, complex(float(T))), Laurent(-order, order))
end

"""
    para_eigvals(H; order=12, nsample=0) -> ParaMatrix

The eigenvalue **functions** (bands) of a **para-Hermitian** Laurent `ParaMatrix`
`H` as a diagonal ParaMatrix (real, ascending `λₖ(θ)`), via per-θ Hermitian `eigvals`
+ DFT-fit. **Differentiable** (eigenvalues are gauge-invariant) — the AD-able
counterpart of `para_eigen` for just the bands. Approximate (raise `order` at
crossings); `@warn`s if `H` is not para-Hermitian.
"""
function para_eigvals(
    H::ParaMatrix{T,S,<:Laurent}; order::Int=12, nsample::Int=0
) where {T,S}
    n = size(H, 1)
    n == size(H, 2) ||
        throw(DimensionMismatch("para_eigvals needs a square H; got $(size(H))"))
    isparahermitian(H) || @warn(
        "para_eigvals: H is not para-Hermitian; using the Hermitian part.", maxlog = 1
    )
    N = nsample > 0 ? nsample : max(64, 16 * order)
    grid = [j / N for j in 0:(N - 1)]
    ev = [eigvals(Hermitian(Matrix(H(t)))) for t in grid]   # real, ascending
    return ParaMatrix(_valfit(ev, n, order, N, complex(float(T))), Laurent(-order, order))
end

# The full para_svd/para_eigen are NOT reverse-mode differentiable (non-smooth
# singular/eigen gauge-fixing). Give a CLEAR error under AD instead of a cryptic
# `llvmcall` failure, pointing at what IS differentiable.
function ChainRulesCore.rrule(
    ::typeof(para_svd), A::ParaMatrix{T,S,<:Laurent}; kw...
) where {T,S}
    return para_svd(A; kw...),
    _ -> error(
        "para_svd is not reverse-mode differentiable: the singular-vector gauge-fixing " *
        "(branch matching + phase normalization) is non-smooth. Use `para_svdvals` for the " *
        "(differentiable) singular-value functions, or `para_qr`/`para_lq` for a fully " *
        "differentiable exact factorization.",
    )
end
function ChainRulesCore.rrule(
    ::typeof(para_eigen), H::ParaMatrix{T,S,<:Laurent}; kw...
) where {T,S}
    return para_eigen(H; kw...),
    _ -> error(
        "para_eigen is not reverse-mode differentiable: the eigenvector gauge-fixing is " *
        "non-smooth. Use `para_eigvals` for the (differentiable) eigenvalue functions, or " *
        "`para_qr`/`para_lq` for a fully differentiable exact factorization.",
    )
end

# --- multivariate (multi-parameter) factorization: unavailable, and genuinely hard ---
# AD status: `spectral_factor` (mutation-free ⇒ ChainRules `cholesky`) and `inv` (rrule,
#   matrix-inverse identity) are differentiable, so `para_qr`/`para_lq` are FULLY
#   AD-transparent (gauge `R`/`L` and para-unitary `Q`), Zygote-validated vs finite diff.
# The full `para_svd`/`para_eigen` are NOT AD (non-smooth singular/eigen gauge-fixing:
#   argmax matching + phase normalization) — differentiating them raises a clear error.
#   The gauge-invariant VALUE functions ARE differentiable: use `para_svdvals` /
#   `para_eigvals`. TODO: an AD-able full para_svd/para_eigen needs a smooth-gauge
#   formulation (open). TODO: ordering convention (analytic vs spectrally-majorised) and
#   crossing passage for `para_svd`/`para_eigen` (see their in-source TODOs).
#
# For ≥2 parameters the parameterized factorizations are unavailable BY DESIGN — it is
# a hard, partly-open problem: in m-D, para-unitary FIR matrices do NOT factor into
# elementary (delay+rotation) lattice blocks (the 1-D lossless lattice is not a
# complete characterization), and there is no multidimensional spectral-factorization
# theorem. Tracked in issue #32. The sampled `eigen`/`svd` DO support ProductClass.
function _multivar_unavailable(fn)
    return error(
        "$fn: parameterized factorization of a MULTI-parameter (ProductClass) object is " *
        "currently unavailable, and it is a difficult (partly open) problem — in ≥2 " *
        "variables para-unitary FIR matrices do not factor into elementary delay+rotation " *
        "blocks and there is no multidimensional spectral-factorization theorem " *
        "[Zhou, Do & Kovačević, IEEE Trans. Image Process. 14(6):760–769, 2005; " *
        "Geronimo & Woerdeman, Ann. of Math. 160(3):839–906, 2004; tracked in issue #32]. " *
        "Use the SAMPLED `eigen`/`svd` (which do support ProductClass), or factorize " *
        "pointwise via `A(p)`.",
    )
end

function spectral_factor(::ParaMatrix{T,S,<:ProductClass}; kw...) where {T,S}
    return _multivar_unavailable("spectral_factor")
end
function para_qr(::ParaMatrix{T,S,<:ProductClass}; kw...) where {T,S}
    return _multivar_unavailable("para_qr")
end
function para_lq(::ParaMatrix{T,S,<:ProductClass}; kw...) where {T,S}
    return _multivar_unavailable("para_lq")
end
function para_svd(::ParaMatrix{T,S,<:ProductClass}; kw...) where {T,S}
    return _multivar_unavailable("para_svd")
end
function para_eigen(::ParaMatrix{T,S,<:ProductClass}; kw...) where {T,S}
    return _multivar_unavailable("para_eigen")
end
function para_svdvals(::ParaMatrix{T,S,<:ProductClass}; kw...) where {T,S}
    return _multivar_unavailable("para_svdvals")
end
function para_eigvals(::ParaMatrix{T,S,<:ProductClass}; kw...) where {T,S}
    return _multivar_unavailable("para_eigvals")
end
