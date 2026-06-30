# Structural algebra laws every faithful matrix algebra must satisfy. Each is
# checked against the *evaluated* (pointwise) LinearAlgebra answer and/or a
# closed form — independent of how the ParaMatrix ring implements *, kron, det, tr.
# (`specmatch` and `randpm` come from test/helpers.jl.)

@testset "determinant is multiplicative: det(A·B) = det(A)·det(B)" begin
    for d in (1, 2, 3), seed in SEEDS
        A = randpm(d, Laurent(-1, 1); seed=seed)
        B = randpm(d, Laurent(-1, 1); seed=seed + 13)
        dAB, dA, dB = det(A * B), det(A), det(B)
        for θ in RNG_PTS
            @test dAB(θ)[1, 1] ≈ dA(θ)[1, 1] * dB(θ)[1, 1] atol = 1e-7   # homomorphism
            @test dAB(θ)[1, 1] ≈ det(A(θ) * B(θ)) atol = 1e-7            # vs pointwise LAPACK
        end
    end
end

@testset "det(Aⁿ) = det(A)ⁿ" begin
    for d in (2, 3), seed in SEEDS, n in (2, 3)
        A = randpm(d, Laurent(-1, 1); seed=seed)
        dAn, dA = det(A^n), det(A)
        for θ in RNG_PTS
            @test dAn(θ)[1, 1] ≈ dA(θ)[1, 1]^n atol = 1e-6
        end
    end
end

@testset "trace is linear + cyclic: tr(A+B)=trA+trB, tr(A·B)=tr(B·A)" begin
    for d in (1, 2, 3, 5), seed in SEEDS
        A = randpm(d, Laurent(-1, 1); seed=seed)
        B = randpm(d, Laurent(-1, 1); seed=seed + 7)
        tAB, tBA, tsum, tA, tB = tr(A * B), tr(B * A), tr(A + B), tr(A), tr(B)
        for θ in RNG_PTS
            @test tsum(θ)[1, 1] ≈ tA(θ)[1, 1] + tB(θ)[1, 1] atol = 1e-10  # linearity
            @test tAB(θ)[1, 1] ≈ tBA(θ)[1, 1] atol = 1e-9                 # cyclic
            @test tAB(θ)[1, 1] ≈ tr(A(θ) * B(θ)) atol = 1e-9             # vs pointwise
        end
    end
end

@testset "Kronecker laws: mixed-product, spectrum, trace, determinant" begin
    for (dA, dB) in ((2, 2), (2, 3), (3, 2)), seed in SEEDS
        A = randpm(dA, Laurent(-1, 1); seed=seed)
        B = randpm(dB, Laurent(-1, 1); seed=seed + 5)
        C = randpm(dA, Laurent(-1, 1); seed=seed + 17)
        D = randpm(dB, Laurent(-1, 1); seed=seed + 29)
        AB = A ⊗ B
        @test AB ≈ kron(A, B)                                            # ⊗ is the kron alias
        mixed = (A ⊗ B) * (C ⊗ D)
        trAB, detAB = tr(AB), det(AB)
        for θ in RNG_PTS
            # mixed-product property: (A⊗B)(C⊗D) = (AC)⊗(BD)
            @test mixed(θ) ≈ kron(A(θ) * C(θ), B(θ) * D(θ)) atol = 1e-8
            # spectrum λ(A⊗B) = { λᵢ·μⱼ } (all pairwise products)
            @test specmatch(
                eigvals(Matrix(AB(θ))),
                vec(eigvals(Matrix(A(θ))) * transpose(eigvals(Matrix(B(θ))))),
            )
            # tr(A⊗B) = trA·trB ,  det(A⊗B) = det(A)^{dB}·det(B)^{dA}
            @test trAB(θ)[1, 1] ≈ tr(A(θ)) * tr(B(θ)) atol = 1e-9
            @test detAB(θ)[1, 1] ≈ det(A(θ))^dB * det(B(θ))^dA atol = 1e-7
        end
    end
end
