# solver/factorizations.jl — QR / LQ / SVD / LU / eigen of a parameterized matrix.
#
# These OVERLOAD the standard `LinearAlgebra` verbs on `ParaMatrix`, so the same
# function names dispatch naturally:  `eigen(A)`, `svd(A)`, `qr(A)`, `lu(A)`,
# `eigvals(A)`, `svdvals(A)`, `pinv(A)`  all work when `A::ParaMatrix`.
#
# SCIENTIFIC NOTE. The QR/SVD/LU/eigendecomposition of A(θ) are algebraic/analytic
# FUNCTIONS of θ — generally NOT members of any finite class — so they are not
# exact `ParaMatrix` factors (unlike `det`, `inv` for para-unitary, `para`, and
# `spectral_factor`, which ARE in-class). Each returns a Para-factorization object
# that is (i) **callable**, `F(p)` = the standard factorization of `A(p)`, and
# (ii) **sampled** on the parameter grid `F.ts` — the circle for ONE parameter, or
# for a multi-parameter `ProductClass` the FULL N-D product grid (then `F.ts` holds
# n-tuples) — with the factor sequences exposed under the usual property names
# (`F.values`, `F.U`, `F.Q`, …). Because the sampled factor at `ts[i]` is *by
# construction* `F(ts[i])`, the identity
#     decompose-then-evaluate   ==   evaluate-then-decompose
# i.e. `eigen(A)(p) == eigen(A(p))`, holds exactly — POINTWISE. Over ≥2 parameters
# there is in general no globally smooth gauge (see `_paramgrid`); use the
# quantum-geometry tools for gauge-invariant geometry over a parameter manifold.

# real, non-negative diagonal phase gauge (continuity / canonical convention)
function _posphase!(Q, R)
    @inbounds for i in 1:min(size(R)...)
        Rii = R[i, i]
        ph = iszero(Rii) ? one(eltype(R)) : Rii / abs(Rii)
        @views Q[:, i] .*= ph
        @views R[i, :] .*= conj(ph)
    end
    return Q, R
end

# Sampling grid for the SAMPLED factorizations. A 1-parameter class is sampled on
# the circle (scalars); a multi-parameter `ProductClass` on the FULL N-D product
# grid (one circle per axis), so each sample point is an n-tuple and the factor is
# the pointwise decomposition there. NOTE (Kato/Rellich): over ≥2 parameters there
# is in general NO globally continuous/smooth choice of eigenvectors/singular
# vectors (degeneracies are codimension-2; a globally smooth gauge need not exist).
# These objects therefore promise POINTWISE correctness only.
_paramgrid(::FunctionClass, nsample::Int) = collect(_circle(nsample))
function _paramgrid(pc::ProductClass, nsample::Int)
    g = _circle(nsample)
    return vec([t for t in Iterators.product(ntuple(_ -> g, length(pc.classes))...)])
end

# ---------- eigen ----------------------------------------------------------
"""
    ParaEigen

Sampled eigendecomposition of a `ParaMatrix` returned by `eigen`; callable
(`F(θ)`) and exposing `F.values`, `F.vectors`, `F.ts`.
"""
struct ParaEigen{PM,F}
    parent::PM
    ts::Vector
    facts::Vector{F}
    herm::Bool
end

"""
    eigen(A::ParaMatrix; nsample=128) -> ParaEigen

Eigendecomposition of `A(θ)` sampled on the circle. `F::ParaEigen` is callable
(`F(θ) == eigen(A(θ))`) and exposes `F.values`, `F.vectors`.

!!! note "Return shape differs from `LinearAlgebra.eigen`"
    `F.values`/`F.vectors` are **sequences over the grid** `F.ts` (a `Vector` of
    per-θ eigenvalue vectors / eigenvector matrices), NOT a single vector/matrix.
    Destructuring `vals, vecs = eigen(A)` therefore yields `Vector{Vector}` /
    `Vector{Matrix}`. For a single point use `eigen(A(θ))`.
"""
# `ishermitian=true` wraps each A(θ) in `Hermitian` so the spectrum is guaranteed
# real (the Hermitian structure flowing into the spectral routine) — use for a
# para-Hermitian A (`isparahermitian(A)`); the flag is stored so the callable agrees.
function LinearAlgebra.eigen(A::ParaMatrix; nsample::Int=128, ishermitian::Bool=false)
    ts = _paramgrid(A.class, nsample)
    wrap = ishermitian ? Hermitian : identity
    return ParaEigen(A, ts, [eigen(wrap(Matrix(A(t)))) for t in ts], ishermitian)
end
function (F::ParaEigen)(θ)
    M = Matrix(getfield(F, :parent)(θ))
    return eigen(getfield(F, :herm) ? Hermitian(M) : M)
end
function Base.getproperty(F::ParaEigen, s::Symbol)
    s === :values && return [f.values for f in getfield(F, :facts)]
    s === :vectors && return [f.vectors for f in getfield(F, :facts)]
    return getfield(F, s)
end
Base.propertynames(::ParaEigen) = (:parent, :ts, :facts, :herm, :values, :vectors)
function Base.iterate(F::ParaEigen, st::Int=1)
    if st == 1
        (F.values, 2)
    elseif st == 2
        (F.vectors, 3)
    else
        nothing
    end
end
Base.length(::ParaEigen) = 2

"""
    eigvals(A::ParaMatrix; nsample=128) -> Vector

The eigenvalue functions sampled on the circle: `out[i] == eigvals(A(θᵢ))` for
`θᵢ` uniform on `[0,1)` (a `Vector` of per-θ eigenvalue vectors). With
`ishermitian=true` each `A(θ)` is wrapped in `Hermitian` ⇒ the bands are real
(use for a para-Hermitian `A`).
"""
function LinearAlgebra.eigvals(A::ParaMatrix; nsample::Int=128, ishermitian::Bool=false)
    wrap = ishermitian ? Hermitian : identity
    return [eigvals(wrap(Matrix(A(t)))) for t in _paramgrid(A.class, nsample)]
end

# ---------- svd ------------------------------------------------------------
"""
    ParaSVD

Sampled SVD of a `ParaMatrix` returned by `svd`; callable (`F(θ)`) and
exposing `F.U`, `F.S`, `F.V`, `F.ts`.
"""
struct ParaSVD{PM,F}
    parent::PM
    ts::Vector
    facts::Vector{F}
end

"""
    svd(A::ParaMatrix; nsample=128) -> ParaSVD

SVD of `A(θ)` sampled on the circle. Callable (`F(θ) == svd(A(θ))`); exposes
`F.U`, `F.S`, `F.V` as **sequences over `F.ts`** (not single matrices — see
`eigen`'s note). Reconstruction: `F.U[i]*Diagonal(F.S[i])*F.V[i]' ≈ A(F.ts[i])`.
"""
function LinearAlgebra.svd(A::ParaMatrix; nsample::Int=128)
    ts = _paramgrid(A.class, nsample)
    return ParaSVD(A, ts, [svd(Matrix(A(t))) for t in ts])
end
(F::ParaSVD)(θ) = svd(Matrix(getfield(F, :parent)(θ)))
function Base.getproperty(F::ParaSVD, s::Symbol)
    s === :U && return [f.U for f in getfield(F, :facts)]
    s === :S && return [f.S for f in getfield(F, :facts)]
    s === :V && return [f.V for f in getfield(F, :facts)]
    s === :Vt && return [f.Vt for f in getfield(F, :facts)]
    return getfield(F, s)
end
Base.propertynames(::ParaSVD) = (:parent, :ts, :facts, :U, :S, :V, :Vt)
function Base.iterate(F::ParaSVD, st::Int=1)
    if st == 1
        (F.U, 2)
    elseif st == 2
        (F.S, 3)
    elseif st == 3
        (F.V, 4)
    else
        nothing
    end
end
Base.length(::ParaSVD) = 3

"""
    svdvals(A::ParaMatrix; nsample=128) -> Vector

The singular-value functions sampled on the circle: `out[i] == svdvals(A(θᵢ))`
for `θᵢ` uniform on `[0,1)`.
"""
function LinearAlgebra.svdvals(A::ParaMatrix; nsample::Int=128)
    return [svdvals(Matrix(A(t))) for t in _paramgrid(A.class, nsample)]
end

# ---------- qr / lq (canonical continuity gauge) ---------------------------
function _qr_gauged(M::AbstractMatrix)
    F = qr(M)
    k = min(size(M)...)
    Q = Matrix(F.Q)[:, 1:k]
    R = Matrix(F.R)[1:k, :]
    _posphase!(Q, R)
    return (; Q, R)
end

"""
    ParaQR

Sampled QR (canonical continuity gauge) of a `ParaMatrix` returned by `qr`;
callable (`F(θ)`) and exposing `F.Q`, `F.R`, `F.ts`.
"""
struct ParaQR{PM<:ParaMatrix,F}
    parent::PM
    ts::Vector
    facts::Vector{F}
end

"""
    qr(A::ParaMatrix; nsample=128) -> ParaQR

QR of `A(θ)` sampled on the circle with the **canonical continuity gauge**
(real, non-negative diagonal of `R`). Callable; exposes `F.Q` (isometries,
`F.Q[i]'F.Q[i] ≈ I`) and `F.R`, with `F.Q[i]*F.R[i] ≈ A(F.ts[i])`.
"""
function LinearAlgebra.qr(A::ParaMatrix; nsample::Int=128)
    ts = _paramgrid(A.class, nsample)
    return ParaQR(A, ts, [_qr_gauged(Matrix(A(t))) for t in ts])
end
(F::ParaQR)(θ) = _qr_gauged(Matrix(getfield(F, :parent)(θ)))
function Base.getproperty(F::ParaQR, s::Symbol)
    s === :Q && return [f.Q for f in getfield(F, :facts)]
    s === :R && return [f.R for f in getfield(F, :facts)]
    return getfield(F, s)
end
Base.propertynames(::ParaQR) = (:parent, :ts, :facts, :Q, :R)
Base.iterate(F::ParaQR, st::Int=1) =
    if st == 1
        (F.Q, 2)
    elseif st == 2
        (F.R, 3)
    else
        nothing
    end
Base.length(::ParaQR) = 2

function _lq_gauged(M::AbstractMatrix)
    F = lq(M)
    L = Matrix(F.L)
    Q = Matrix(F.Q)
    @inbounds for i in 1:min(size(L)...)
        Lii = L[i, i]
        ph = iszero(Lii) ? one(eltype(L)) : Lii / abs(Lii)
        @views L[:, i] .*= conj(ph)
        @views Q[i, :] .*= ph
    end
    return (; L, Q)
end

"""
    ParaLQ

Sampled LQ (canonical continuity gauge) of a `ParaMatrix` returned by `lq`;
callable (`F(θ)`) and exposing `F.L`, `F.Q`, `F.ts`.
"""
struct ParaLQ{PM<:ParaMatrix,F}
    parent::PM
    ts::Vector
    facts::Vector{F}
end

"""
    lq(A::ParaMatrix; nsample=128) -> ParaLQ

LQ of `A(θ)` sampled on the circle (canonical gauge; `F.Q[i]` isometric,
`F.L[i]*F.Q[i] ≈ A(F.ts[i])`).
"""
function LinearAlgebra.lq(A::ParaMatrix; nsample::Int=128)
    ts = _paramgrid(A.class, nsample)
    return ParaLQ(A, ts, [_lq_gauged(Matrix(A(t))) for t in ts])
end
(F::ParaLQ)(θ) = _lq_gauged(Matrix(getfield(F, :parent)(θ)))
function Base.getproperty(F::ParaLQ, s::Symbol)
    s === :L && return [f.L for f in getfield(F, :facts)]
    s === :Q && return [f.Q for f in getfield(F, :facts)]
    return getfield(F, s)
end
Base.propertynames(::ParaLQ) = (:parent, :ts, :facts, :L, :Q)
Base.iterate(F::ParaLQ, st::Int=1) =
    if st == 1
        (F.L, 2)
    elseif st == 2
        (F.Q, 3)
    else
        nothing
    end
Base.length(::ParaLQ) = 2

# ---------- lu -------------------------------------------------------------
"""
    ParaLU

Sampled LU (partial pivoting) of a `ParaMatrix` returned by `lu`; callable
(`F(θ)`) and exposing `F.L`, `F.U`, `F.p`, `F.ts`.
"""
struct ParaLU{PM<:ParaMatrix,F}
    parent::PM
    ts::Vector
    facts::Vector{F}
end

"""
    lu(A::ParaMatrix; nsample=128) -> ParaLU

LU (partial pivoting) of `A(θ)` sampled on the circle. Callable; exposes
`F.L`, `F.U`, `F.p`, with `F.L[i]*F.U[i] ≈ A(F.ts[i])[F.p[i], :]`.
"""
function LinearAlgebra.lu(A::ParaMatrix; nsample::Int=128, check::Bool=true)
    ts = _paramgrid(A.class, nsample)
    return ParaLU(A, ts, [lu(Matrix(A(t)); check=check) for t in ts])
end
(F::ParaLU)(θ) = lu(Matrix(getfield(F, :parent)(θ)))
function Base.getproperty(F::ParaLU, s::Symbol)
    s === :L && return [f.L for f in getfield(F, :facts)]
    s === :U && return [f.U for f in getfield(F, :facts)]
    s === :p && return [f.p for f in getfield(F, :facts)]
    return getfield(F, s)
end
Base.propertynames(::ParaLU) = (:parent, :ts, :facts, :L, :U, :p)
function Base.iterate(F::ParaLU, st::Int=1)
    if st == 1
        (F.L, 2)
    elseif st == 2
        (F.U, 3)
    elseif st == 3
        (F.p, 4)
    else
        nothing
    end
end
Base.length(::ParaLU) = 3

# ---------- polar (no stdlib verb; new export) -----------------------------
"""
    ParaPolar

Sampled polar decomposition of a `ParaMatrix` returned by [`polar`](@ref); callable
(`F(θ)`) and exposing `F.U`, `F.P`, `F.ts`.
"""
struct ParaPolar{PM<:ParaMatrix,F}
    parent::PM
    ts::Vector
    facts::Vector{F}
end

function _polar(M::AbstractMatrix)
    F = svd(M)
    return (; U=(F.U * F.Vt), P=(F.V * Diagonal(F.S) * F.Vt))
end

"""
    polar(A::ParaMatrix; nsample=128) -> ParaPolar

Polar factors `A(θ) = U(θ)P(θ)` sampled on the circle: `F.U[i]` is the
(para-)unitary gauge and `F.P[i] ⪰ 0` Hermitian. For square or tall `A` (`m ≥ n`)
`F.U[i]'F.U[i] ≈ I` (column isometry); for wide `A` it is `F.U[i]*F.U[i]' ≈ I`.
"""
function polar(A::ParaMatrix; nsample::Int=128)
    ts = _paramgrid(A.class, nsample)
    return ParaPolar(A, ts, [_polar(Matrix(A(t))) for t in ts])
end
(F::ParaPolar)(θ) = _polar(Matrix(getfield(F, :parent)(θ)))
function Base.getproperty(F::ParaPolar, s::Symbol)
    s === :U && return [f.U for f in getfield(F, :facts)]
    s === :P && return [f.P for f in getfield(F, :facts)]
    return getfield(F, s)
end
Base.propertynames(::ParaPolar) = (:parent, :ts, :facts, :U, :P)
Base.iterate(F::ParaPolar, st::Int=1) =
    if st == 1
        (F.U, 2)
    elseif st == 2
        (F.P, 3)
    else
        nothing
    end
Base.length(::ParaPolar) = 2

# ---------- regularized pseudo-inverse (SVD divergence removal) -------------
"""
    pinv(A::ParaMatrix; atol=0, rtol=…, nsample=128) -> (ts, Aplus)

Regularized Moore–Penrose pseudo-inverse of `A(θ)` sampled on the circle, via the
truncated SVD (singular values ≤ `max(atol, rtol·σ₁)` are dropped). This is the
**SVD divergence removal**: near-zero singular directions are projected out rather
than inverted, so `Aplus[i]` stays bounded even where `A(ts[i])` is (near-)singular.
"""
function LinearAlgebra.pinv(
    A::ParaMatrix{T}; atol::Real=0, rtol::Real=0, nsample::Int=128
) where {T}
    rt = rtol > 0 ? rtol : (atol > 0 ? 0.0 : eps(real(float(one(T)))) * minimum(size(A)))
    ts = _paramgrid(A.class, nsample)
    return ts, [pinv(Matrix(A(t)); atol=atol, rtol=rt) for t in ts]
end

# ---------- BlockParaMatrix (independently-parameterized blocks) ------------
# Sampled pointwise over the n-D grid of the block's INDEPENDENT parameters; the
# same per-point-only guarantee applies (no global gauge over ≥2 parameters).
function _paramgrid_n(n::Int, nsample::Int)
    n == 1 && return collect(_circle(nsample))
    g = _circle(nsample)
    return vec([t for t in Iterators.product(ntuple(_ -> g, n)...)])
end

"""
    eigen(M::BlockParaMatrix; nsample=24, ishermitian=false) -> ParaEigen

Eigendecomposition of `M` sampled on the grid of its independent parameters;
callable (`F(p) == eigen(M(p))`), `F.values`/`F.vectors` are sequences over `F.ts`.
"""
function LinearAlgebra.eigen(M::BlockParaMatrix; nsample::Int=24, ishermitian::Bool=false)
    ts = _paramgrid_n(M.nparams, nsample)
    wrap = ishermitian ? Hermitian : identity
    return ParaEigen(M, ts, [eigen(wrap(Matrix(M(t)))) for t in ts], ishermitian)
end
function LinearAlgebra.eigvals(M::BlockParaMatrix; nsample::Int=24, ishermitian::Bool=false)
    wrap = ishermitian ? Hermitian : identity
    return [eigvals(wrap(Matrix(M(t)))) for t in _paramgrid_n(M.nparams, nsample)]
end

"""
    svd(M::BlockParaMatrix; nsample=24) -> ParaSVD

SVD of `M` sampled on the grid of its independent parameters; callable
(`F(p) == svd(M(p))`), exposing `F.U`/`F.S`/`F.V` as sequences over `F.ts`.
"""
function LinearAlgebra.svd(M::BlockParaMatrix; nsample::Int=24)
    ts = _paramgrid_n(M.nparams, nsample)
    return ParaSVD(M, ts, [svd(Matrix(M(t))) for t in ts])
end
function LinearAlgebra.svdvals(M::BlockParaMatrix; nsample::Int=24)
    return [svdvals(Matrix(M(t))) for t in _paramgrid_n(M.nparams, nsample)]
end
