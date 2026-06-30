# Block-matrix construction + variadic combinators added in this patch:
# the `[A B; C D]` literal (hvcat), and `⊗`/`⊕` over more than two arguments.
# Checked pointwise against the assembled dense block matrix.

@testset "hvcat block literal [A B; C D]" begin
    for d in (1, 2, 3), seed in SEEDS
        A = randpm(d, Laurent(-1, 1); seed=seed)
        B = randpm(d, Laurent(-1, 1); seed=seed + 1)
        C = randpm(d, Laurent(-1, 1); seed=seed + 2)
        D = randpm(d, Laurent(-1, 1); seed=seed + 3)
        M = [A B; C D]
        @test M == hvcat((2, 2), A, B, C, D)             # literal lowers to hvcat
        @test [A B] == hcat(A, B)                         # row literal == hcat
        Z = zero(A)
        @test [A Z; Z B] == (A ⊕ B)                       # block-diagonal literal == direct sum
        for θ in RNG_PTS
            @test M(θ) ≈ [A(θ) B(θ); C(θ) D(θ)] atol = 1e-12
        end
    end
end

@testset "variadic ⊗ and ⊕ fold left" begin
    for seed in SEEDS
        A = randpm(2, Laurent(-1, 1); seed=seed)
        B = randpm(2, Laurent(-1, 1); seed=seed + 1)
        C = randpm(2, Laurent(-1, 1); seed=seed + 2)
        @test (A ⊗ B ⊗ C) == kron(kron(A, B), C)
        @test (A ⊕ B ⊕ C) == directsum(directsum(A, B), C)
        for θ in RNG_PTS
            @test (A ⊗ B ⊗ C)(θ) ≈ kron(kron(A(θ), B(θ)), C(θ)) atol = 1e-10
            @test (A ⊕ B ⊕ C)(θ) ≈ cat(A(θ), B(θ), C(θ); dims=(1, 2)) atol = 1e-12
        end
    end
end
