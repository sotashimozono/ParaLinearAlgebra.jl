# Multi-parameter (N-D) decomposition: a ProductClass is sampled on the FULL N-D
# product grid and decomposed pointwise. Oracle = a separable object with a KNOWN
# spectrum on the torus, plus pointwise reconstruction and callable consistency.
# (Over ≥2 parameters only POINTWISE correctness is promised — no global gauge.)

# build A(s,t) = diag(z_s, z_t),  z_s = e^{2πis},  z_t = e^{2πit}  (a 2-torus character)
function _char2()
    pc = ProductClass(Laurent(-1, 1), Laurent(-1, 1))
    pw = powers(pc)
    idx(a, b) = findfirst(==(CartesianIndex(a, b)), pw)
    coeffs = [zeros(ComplexF64, 2, 2) for _ in 1:nbasis(pc)]
    coeffs[idx(1, 0)][1, 1] = 1      # entry (1,1) = z_s
    coeffs[idx(0, 1)][2, 2] = 1      # entry (2,2) = z_t
    return ParaMatrix(coeffs, pc)
end

@testset "N-D eigen: 2-torus character, known spectrum {z_s, z_t}" begin
    A = _char2()
    @test A((0.2, 0.3)) ≈ ComplexF64[cispi(0.4) 0; 0 cispi(0.6)]
    F = eigen(A; nsample=6)
    @test length(F.ts) == 6^2
    @test all(p -> p isa Tuple && length(p) == 2, F.ts)        # grid points are 2-tuples
    for (p, vals) in zip(F.ts, F.values)
        s, t = p
        @test specmatch(vals, [cispi(2s), cispi(2t)])          # eigenvalues = {z_s, z_t}
    end
    for p in ((0.1, 0.7), (0.45, 0.45), (0.0, 0.5))            # callable == pointwise eigen
        @test specmatch(F(p).values, eigvals(A(p)))
    end
end

@testset "N-D svd: unitary character ⇒ σ ≡ 1; pointwise reconstruction" begin
    A = _char2()
    S = svd(A; nsample=5)
    @test length(S.ts) == 5^2
    for sv in S.S
        @test all(≈(1), sv)                                    # unitary ⇒ all singular values 1
    end
    for (p, U, s, V) in zip(S.ts, S.U, S.S, S.V)
        @test U * Diagonal(s) * V' ≈ A(p) atol = 1e-10         # reconstruction at each grid point
    end
end

@testset "N-D eigvals/svdvals and 3-parameter grid sizes" begin
    pc = ProductClass(Laurent(-1, 1), Polynomial(1))
    A = ParaMatrix([randn(MersenneTwister(i), ComplexF64, 3, 3) for i in 1:nbasis(pc)], pc)
    @test length(eigvals(A; nsample=4)) == 4^2
    @test length(svdvals(A; nsample=4)) == 4^2
    # a 3-parameter object samples an nsample³ grid
    pc3 = ProductClass(Laurent(-1, 1), Laurent(-1, 1), Laurent(-1, 1))
    A3 = ParaMatrix(
        [randn(MersenneTwister(i), ComplexF64, 2, 2) for i in 1:nbasis(pc3)], pc3
    )
    @test length(eigvals(A3; nsample=3)) == 3^3
    @test all(p -> length(p) == 3, eigen(A3; nsample=3).ts)
end
