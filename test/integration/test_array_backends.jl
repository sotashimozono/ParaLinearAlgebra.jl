# tests the ARRAY-BACKEND genericity contract of the coefficient blocks: a block
# may be ANY `AbstractMatrix{<:BlasFloat}`, not only `Base.Matrix`. The whole
# parameterized algebra AND the factorizations must accept such blocks and compute
# correct results. (Computed/factorized blocks may come back as dense `Matrix` —
# correctness is guaranteed, exotic block-type PRESERVATION is not.)
#
# This is neutral, backend-AGNOSTIC hardening: it names no downstream, but it is
# exactly what lets another package feed its own matrix type (e.g. a matricized
# tensor) as coefficient blocks and get the parameterized algebra + canonicalization
# (`para_qr`) for free. Sparse / Static / BigFloat blocks are already covered for the
# CORE algebra in test/core/test_paramatrix.jl; here we (a) add the FACTORIZATION
# path for non-`Array` blocks and (b) stress the hardest case with `OpaqueMat`.
#
# `OpaqueMat` is a deliberately minimal AbstractMatrix — it implements only the
# AbstractArray interface, with NO LinearAlgebra specializations. If any code path
# secretly assumed `Base.Matrix`, it surfaces here as an error, not a silent dense
# fallback. (`include` runs at module global scope, so this struct is top-level.)

struct OpaqueMat{T} <: AbstractMatrix{T}
    data::Matrix{T}
end
Base.size(A::OpaqueMat) = size(A.data)
Base.getindex(A::OpaqueMat, i::Int, j::Int) = A.data[i, j]
Base.setindex!(A::OpaqueMat, v, i::Int, j::Int) = (A.data[i, j] = v)
# no `similar` override: Base's default `similar(::AbstractArray, T, dims)` returns a
# correctly-dimensioned `Array` (so `tr`/`diag`/matmul densify to Matrix) — which is
# exactly the "results may come back dense" half of the contract.

function _opaque(d, cls; seed=0)
    return ParaMatrix(
        [
            OpaqueMat(randn(MersenneTwister(seed + 137i), ComplexF64, d, d)) for
            i in 1:nbasis(cls)
        ],
        cls,
    )
end
# same coefficient data, but plain-Matrix blocks — the oracle to compare against
_denseof(A) = ParaMatrix([Matrix(c) for c in coefficients(A)], function_class(A))

@testset "opaque AbstractMatrix blocks: construct + core algebra vs Matrix oracle" begin
    for cls in (Laurent(-2, 2), Analytic(2), Polynomial(3)),
        d in (1, 2, 4),
        seed in (11, 47)

        A = _opaque(d, cls; seed=seed)
        B = _opaque(d, cls; seed=seed + 1)
        @test eltype(A) == ComplexF64                        # eltype from the block, not Matrix
        @test coefficients(A)[1] isa OpaqueMat               # input blocks kept as given
        rA, rB = _denseof(A), _denseof(B)
        for θ in RNG_PTS
            @test Matrix(A(θ)) ≈ rA(θ)                       # inherited evaluate
            @test Matrix((A + B)(θ)) ≈ rA(θ) + rB(θ)
            @test Matrix((A - B)(θ)) ≈ rA(θ) - rB(θ)
            @test Matrix((2.5 * A)(θ)) ≈ 2.5 * rA(θ)
            @test Matrix((A * B)(θ)) ≈ rA(θ) * rB(θ)         # convolution over opaque blocks
            @test Matrix((A ⊗ B)(θ)) ≈ kron(rA(θ), rB(θ))
            @test Matrix((A ⊕ B)(θ)) ≈ cat(rA(θ), rB(θ); dims=(1, 2))
            @test Matrix((A^2)(θ)) ≈ rA(θ)^2
            @test Matrix(transpose(A)(θ)) ≈ transpose(rA(θ))
            @test tr(A)(θ)[1, 1] ≈ tr(rA(θ))
        end
        @test norm(A, 2) ≈ norm(rA, 2)
    end
end

@testset "opaque blocks: Laurent para-structure (para / adjoint / det / predicates)" begin
    for d in (2, 3), seed in (23, 71)
        A = _opaque(d, Laurent(-2, 2); seed=seed)
        rA = _denseof(A)
        for θ in RNG_PTS
            @test Matrix(para(A)(θ)) ≈ adjoint(rA(θ))
            @test Matrix(A'(θ)) ≈ adjoint(rA(θ))
            @test det(A)(θ)[1, 1] ≈ det(rA(θ))
        end
        @test isparahermitian(parahermitianpart(A))
    end
    # mixed block types: an opaque ParaMatrix combined with a Matrix-block `paraeye`
    # (a downstream adds identities/shifts built from plain Matrix). Must not error.
    A = _opaque(3, Laurent(-1, 1); seed=5)
    G = para(A) * A + paraeye(3, ComplexF64, Laurent(-2, 2))
    @test ishermitian(G) && ispositive(G)
end

@testset "opaque blocks: factorizations densify correctly (the canonicalization path)" begin
    for d in (2, 3), seed in (11, 71)
        A = _opaque(d, Laurent(-1, 1); seed=seed)
        rA = _denseof(A)
        # sampled spectra match the Matrix-block oracle (LAPACK path densifies input)
        @test specmatch(_sortc(reduce(vcat, eigvals(A))), _sortc(reduce(vcat, eigvals(rA))))
        @test sort(reduce(vcat, svdvals(A))) ≈ sort(reduce(vcat, svdvals(rA)))
        # spectral factor of an opaque-sourced PD para-Hermitian Gram: M·para(M) ≈ G
        M0 = _opaque(d, Analytic(1); seed=seed + 3)
        G = para(M0) * M0 + paraeye(d, ComplexF64, Laurent(-1, 1))
        @test ishermitian(G) && ispositive(G)
        W = spectral_factor(G; N=32)
        for θ in (0.1, 0.4, 0.7, 0.95)
            @test W(θ) * para(W)(θ) ≈ G(θ) atol = 1e-6
        end
        # exact parameterized → parameterized canonicalization straight off opaque sources
        Fqr = para_qr(G; N=32, order=40)
        Flq = para_lq(G; N=32, order=40)
        @test Fqr.residual < 1e-6 && Flq.residual < 1e-6
        for θ in (0.2, 0.6)
            @test G(θ) ≈ Fqr.Q(θ) * Fqr.R(θ) atol = 1e-6
            @test G(θ) ≈ Flq.L(θ) * Flq.Q(θ) atol = 1e-6
        end
    end
end

# StaticArrays already cover the CORE algebra (test_paramatrix.jl); pin the
# factorization path for SMatrix blocks too — what a small-tensor backend would use.
@testset "SMatrix blocks flow through the factorization path" begin
    ct = [
        SMatrix{3,3,ComplexF64}(randn(MersenneTwister(9 + 7i), ComplexF64, 3, 3)) for
        i in 1:3
    ]
    A = ParaMatrix(ct, Laurent(-1, 1))
    @test coefficients(A)[1] isa SMatrix
    @test specmatch(
        _sortc(reduce(vcat, eigvals(A))), _sortc(reduce(vcat, eigvals(_denseof(A))))
    )
    G = para(A) * A + paraeye(3, ComplexF64, Laurent(-2, 2))   # para(A)*A lives in Laurent(-2,2)
    @test para_qr(G; N=32, order=40).residual < 1e-6
end
