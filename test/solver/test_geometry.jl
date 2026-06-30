# Quantum-geometry oracles. The Qi–Wu–Zhang / massive-Dirac model is itself a
# Laurent⊗Laurent ParaMatrix (sin/cos = (z±z⁻¹)/…), so we validate the Chern
# number against its KNOWN topological phase diagram (Fukui–Hatsugai–Suzuki), the
# Berry curvature against the closed-form 2-level d-vector formula, and the two
# geometry methods (QGT sum-over-states vs FHS plaquette) against each other —
# all independent of the library's own decomposition machinery.

const _σx = ComplexF64[0 1; 1 0]
const _σy = ComplexF64[0 -im; im 0]
const _σz = ComplexF64[1 0; 0 -1]

# H(kx,ky) = sin kx σx + sin ky σy + (m + cos kx + cos ky) σz,  kx=2πs, ky=2πt
function qwz(m)
    pc = ProductClass(Laurent(-1, 1), Laurent(-1, 1))
    pw = powers(pc)
    idx(a, b) = findfirst(==(CartesianIndex(a, b)), pw)
    C = [zeros(ComplexF64, 2, 2) for _ in 1:nbasis(pc)]
    C[idx(0, 0)] .= m .* _σz
    C[idx(1, 0)] .= _σx ./ (2im) .+ _σz ./ 2     # e^{+ikx}: sin/(2i) σx + cos/2 σz
    C[idx(-1, 0)] .= -_σx ./ (2im) .+ _σz ./ 2
    C[idx(0, 1)] .= _σy ./ (2im) .+ _σz ./ 2
    C[idx(0, -1)] .= -_σy ./ (2im) .+ _σz ./ 2
    return ParaMatrix(C, pc)
end
_dvec(m, s, t) = [sinpi(2s), sinpi(2t), m + cospi(2s) + cospi(2t)]

@testset "QWZ Hamiltonian is Hermitian and equals d·σ" begin
    for m in (0.5, 1.5, -1.0), (s, t) in ((0.1, 0.2), (0.37, 0.6), (0.5, 0.25))
        H = qwz(m)((s, t))
        @test H ≈ H'                                         # Hermitian on the torus
        d = _dvec(m, s, t)
        @test H ≈ d[1] * _σx + d[2] * _σy + d[3] * _σz atol = 1e-12
    end
end

@testset "Chern number vs the known QWZ phase diagram (FHS, gauge-invariant)" begin
    @test chern_number(qwz(1.0); nsample=18) == 1            # 0 < m < 2  : C = +1
    @test chern_number(qwz(-1.0); nsample=18) == -1          # -2 < m < 0 : C = -1
    @test chern_number(qwz(3.0); nsample=18) == 0            # m > 2      : trivial
    @test chern_number(qwz(-3.0); nsample=18) == 0           # m < -2     : trivial
    @test chern_number(qwz(1.0); band=2, nsample=18) == -1   # the two bands carry opposite C
    # FHS is integer-exact on coarse meshes ⇒ coarse and fine grids agree
    @test chern_number(qwz(1.0); nsample=10) == 1
    @test chern_number(qwz(1.0); nsample=33) == 1
end

@testset "QGT is Hermitian PSD; metric symmetric; Berry curvature vs d-vector formula" begin
    for m in (0.7, 1.4), (s, t) in ((0.13, 0.42), (0.6, 0.27))
        H = qwz(m)
        Q = quantum_geometric_tensor(H, (s, t); band=1)
        @test Q ≈ Q'                                         # Hermitian
        @test minimum(real, eigvals(Hermitian(Q))) ≥ -1e-9   # PSD (Provost–Vallée)
        g = fubini_study_metric(H, (s, t); band=1)
        @test g ≈ g' && g ≈ real(Q)                          # metric = Re Q, symmetric
        # closed-form 2-level Berry curvature: ½ d̂·(∂_s d̂ × ∂_t d̂)  (magnitude)
        dh(a, b) = (v=_dvec(m, a, b); v / norm(v))
        h = 1e-6
        ds = (dh(s + h, t) - dh(s - h, t)) / 2h
        dt = (dh(s, t + h) - dh(s, t - h)) / 2h
        F_closed = 0.5 * dot(dh(s, t), cross(ds, dt))
        @test abs(berry_curvature(H, (s, t); band=1)[1, 2]) ≈ abs(F_closed) rtol = 1e-3
    end
end

@testset "two independent methods agree: (1/2π)∮ berry_curvature == chern_number" begin
    for m in (1.0, -1.0, 1.6)
        H = qwz(m)
        N = 40
        gg = _circle_pts(N)
        F = sum(berry_curvature(H, (s, t); band=1)[1, 2] for s in gg, t in gg) / N^2
        @test round(Int, F / (2π)) == chern_number(H; band=1, nsample=24)
    end
end
