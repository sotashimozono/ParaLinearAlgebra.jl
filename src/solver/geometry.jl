# solver/geometry.jl — quantum geometry of a parameterized HERMITIAN matrix over
# its parameter manifold. For H(p₁,…,pₙ) a ParaMatrix that is Hermitian at every p
# (a parameterized Hamiltonian / Bloch matrix), this gives:
#   • the quantum geometric tensor Q (Provost–Vallée 1980): Re Q = Fubini–Study
#     metric, Im Q ∝ Berry curvature — computed from ∂H via sum-over-states, so
#     NO numerical eigenvector derivatives are needed;
#   • the first Chern number over a 2-torus by the gauge-invariant Fukui–Hatsugai–
#     Suzuki lattice method (JPSJ 74, 1674 (2005)) — robust because it needs no
#     eigenvector gauge fixing (which is in general impossible over ≥2 parameters).

_nparams(A::ParaMatrix) = _nparams(A.class)
_nparams(::FunctionClass) = 1
_nparams(pc::ProductClass) = length(pc.classes)

# ∂_μ H at p: the 2-arg derivative for one parameter, the per-axis form otherwise
function _partial(H::ParaMatrix, p, μ::Int)
    return _nparams(H) == 1 ? evaluate_deriv(H, p) : evaluate_deriv(H, p, μ)
end

"""
    quantum_geometric_tensor(H, p; band=1) -> Matrix

The quantum geometric tensor `Q_{μν}` of the `band`-th eigenstate of the Hermitian
`H(p)` at the parameter point `p`, via the sum-over-states form (Provost–Vallée,
*Commun. Math. Phys.* 76 (1980)):

```
Q_{μν} = Σ_{m≠n} ⟨n|∂_μH|m⟩⟨m|∂_νH|n⟩ / (Eₙ − Eₘ)²
```

using the analytic parameter derivatives `∂_μH` (so no numerical differentiation
of eigenvectors). `Re Q` is the Fubini–Study (quantum) metric [`fubini_study_metric`](@ref);
`Im Q` gives the Berry curvature [`berry_curvature`](@ref) (sign convention there). `p` is a scalar for a
one-parameter `H`, an n-tuple for a `ProductClass`. Bands are ordered as by `eigen`.
"""
function quantum_geometric_tensor(H::ParaMatrix, p; band::Int=1)
    nax = _nparams(H)
    F = eigen(Hermitian(Matrix(H(p))))
    d = size(H, 1)
    1 ≤ band ≤ d || throw(ArgumentError("band=$band out of range 1:$d"))
    ψn, En = F.vectors[:, band], F.values[band]
    dH = ntuple(μ -> _partial(H, p, μ), nax)
    Q = zeros(ComplexF64, nax, nax)
    for m in 1:d
        m == band && continue
        gap = En - F.values[m]
        ψm = @view F.vectors[:, m]
        a = ntuple(μ -> dot(ψn, dH[μ] * ψm), nax)        # aμ = ⟨n|∂_μH|m⟩
        for μ in 1:nax, ν in 1:nax
            Q[μ, ν] += a[μ] * conj(a[ν]) / gap^2          # ⟨m|∂_νH|n⟩ = conj(aν)
        end
    end
    return Q
end

"""
    fubini_study_metric(H, p; band=1) -> Matrix

The Fubini–Study (quantum) metric `g_{μν} = Re Q_{μν}` — a real symmetric PSD
matrix — of the `band`-th eigenstate at `p`. See [`quantum_geometric_tensor`](@ref).
"""
function fubini_study_metric(H::ParaMatrix, p; band::Int=1)
    return real(quantum_geometric_tensor(H, p; band=band))
end

"""
    berry_curvature(H, p; band=1) -> Matrix

The Berry curvature `F_{μν} = 2 Im Q_{μν}` (a real antisymmetric matrix) of the
`band`-th eigenstate at `p`; for two parameters the physical curvature is the
`[1,2]` entry. See [`quantum_geometric_tensor`](@ref).

The overall sign of the Berry curvature is a well-known convention freedom (some
references write `−2 Im Q`, Berry connection `A_μ = i⟨ψ|∂_μψ⟩`). We fix it the
other way (`A_μ = −i⟨ψ|∂_μψ⟩`) so that the library is *internally consistent*:
`(1/2π) ∬ F over the torus == chern_number` (the gauge-invariant FHS value).
"""
function berry_curvature(H::ParaMatrix, p; band::Int=1)
    return 2 .* imag(quantum_geometric_tensor(H, p; band=band))
end

"""
    chern_number(H; band=1, nsample=24) -> Int

The first Chern number of the `band`-th eigenstate of the Hermitian two-parameter
`H` over the torus `[0,1)²`, by the **gauge-invariant** Fukui–Hatsugai–Suzuki
lattice method (*J. Phys. Soc. Jpn.* 74 (2005) 1674): U(1) link variables between
neighbouring grid eigenvectors and the log of the plaquette holonomy, summed and
divided by `2πi`. Returns the integer directly; robust on coarse grids and needs
no eigenvector gauge fixing. `H` must be a 2-parameter `ProductClass` and Hermitian
on the torus (gapped at `band` — away from degeneracies).
"""
function chern_number(H::ParaMatrix; band::Int=1, nsample::Int=24)
    _nparams(H) == 2 || throw(
        ArgumentError("chern_number needs a 2-parameter H (ProductClass of two axes)")
    )
    N = nsample
    g = _circle(N)
    ψ = Matrix{Vector{ComplexF64}}(undef, N, N)
    for i in 1:N, j in 1:N
        ψ[i, j] = eigen(Hermitian(Matrix(H((g[i], g[j]))))).vectors[:, band]
    end
    link(a, b) = (z=dot(a, b); z / abs(z))             # normalised U(1) link variable
    w(i) = mod1(i, N)                                     # periodic wrap on the torus
    total = 0.0im
    for i in 1:N, j in 1:N
        U1 = link(ψ[i, j], ψ[w(i + 1), j])               # k → k+1̂
        U2 = link(ψ[w(i + 1), j], ψ[w(i + 1), w(j + 1)]) # k+1̂ → k+1̂+2̂
        U3 = link(ψ[w(i + 1), w(j + 1)], ψ[i, w(j + 1)]) # k+1̂+2̂ → k+2̂
        U4 = link(ψ[i, w(j + 1)], ψ[i, j])               # k+2̂ → k
        total += log(U1 * U2 * U3 * U4)                  # plaquette field strength
    end
    return round(Int, real(total / (2π * im)))
end
