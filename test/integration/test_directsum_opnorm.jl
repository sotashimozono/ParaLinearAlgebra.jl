# Oracles for the operations added in this patch: scalar division `/`, the H∞
# sup-norm `opnorm`, and the direct sum `⊕` (companion of `⊗`). Checked against
# closed forms / pointwise LinearAlgebra / structural inequalities.

@testset "scalar division A/α" begin
    for d in (1, 2, 3), seed in SEEDS
        A = randpm(d, Laurent(-2, 2); seed=seed)
        for α in (2.0, -3.0, 1.5im, 0.4 * cispi(0.2))
            @test ((A / α) * α) ≈ A                       # undoes scalar multiplication
            @test (A / α) ≈ (1 / α) * A                    # = (1/α)·A
            for θ in RNG_PTS
                @test (A / α)(θ) ≈ A(θ) / α atol = 1e-12   # pointwise
            end
        end
    end
end

@testset "opnorm = H∞ sup-norm max_θ ‖A(θ)‖₂" begin
    Id = ComplexF64[1 0; 0 1]
    J = ComplexF64[0 -1; 1 0]
    R = ParaMatrix(
        [0.5Id + 0.5im * J, zeros(ComplexF64, 2, 2), 0.5Id - 0.5im * J], Laurent(-1, 1)
    )
    @test opnorm(R) ≈ 1 atol = 1e-6                                 # para-unitary ⇒ all σ = 1
    D = ParaMatrix([ComplexF64[1 0; 0 0], ComplexF64[0 0; 0 1]], Analytic(1))  # diag(1, z)
    @test opnorm(D) ≈ 1 atol = 1e-6
    for α in (2.0, 1.5im, -3.0)
        @test opnorm(α * paraeye(3, ComplexF64, Laurent(-1, 1))) ≈ abs(α) atol = 1e-6  # |α|·I
    end
    for d in (2, 3), seed in SEEDS
        A = randpm(d, Laurent(-1, 1); seed=seed)
        B = randpm(d, Laurent(-1, 1); seed=seed + 9)
        @test opnorm(A + B) ≤ opnorm(A) + opnorm(B) + 1e-6          # triangle inequality
        @test opnorm(A * B) ≤ opnorm(A) * opnorm(B) + 1e-6          # submultiplicative
        fine = maximum(opnorm(Matrix(A(t))) for t in range(0, 1; length=4097)[1:4096])
        @test opnorm(A) ≈ fine rtol = 1e-3                          # default grid sup is converged
    end
end

@testset "direct sum A ⊕ B: spectrum/det/trace compose" begin
    for (dA, dB) in ((1, 2), (2, 2), (2, 3), (3, 1)), seed in SEEDS
        A = randpm(dA, Laurent(-1, 1); seed=seed)
        B = randpm(dB, Laurent(-1, 1); seed=seed + 11)
        S = A ⊕ B
        @test S == directsum(A, B)                                  # infix alias
        @test size(S) == (dA + dB, dA + dB)
        dS, tS = det(S), tr(S)
        for θ in RNG_PTS
            @test S(θ) ≈ cat(A(θ), B(θ); dims=(1, 2)) atol = 1e-12  # block diagonal
            @test specmatch(
                eigvals(Matrix(S(θ))), vcat(eigvals(Matrix(A(θ))), eigvals(Matrix(B(θ))))
            )                                                       # eig(A⊕B) = eig A ∪ eig B
            @test dS(θ)[1, 1] ≈ det(A(θ)) * det(B(θ)) atol = 1e-7  # det multiplies
            @test tS(θ)[1, 1] ≈ tr(A(θ)) + tr(B(θ)) atol = 1e-10   # trace adds
        end
    end
end
