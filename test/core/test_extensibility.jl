# tests the AbstractParaMatrix extension contract (src/core/paramatrix.jl): a
# *user-defined* subtype that implements only the small documented interface
# (`coefficients` + `function_class`, and optionally `_rebuild`) must inherit the
# entire algebra AND the factorizations for free. This is the executable spec
# behind the "extension-friendly" design — the core logic dispatches on
# AbstractParaMatrix, never on the concrete ParaMatrix.
#
# `include` runs a test file at the module's global scope, so these `struct`s are
# top-level (legal) even though the include sits inside a @testset.

# (1) a TYPE-PRESERVING subtype: carries extra metadata (a tag) and overrides
# `_rebuild`, so every operation returns a TaggedPara that keeps the tag.
struct TaggedPara{T,S<:AbstractMatrix{T},C<:FunctionClass} <: AbstractParaMatrix{T,S,C}
    coeffs::Vector{S}
    class::C
    tag::String
end
ParaLinearAlgebra.coefficients(A::TaggedPara) = A.coeffs
ParaLinearAlgebra.function_class(A::TaggedPara) = A.class
ParaLinearAlgebra._rebuild(A::TaggedPara, coeffs, class) = TaggedPara(coeffs, class, A.tag)

# (2) a MINIMAL subtype: implements ONLY the two required methods (no `_rebuild`,
# no `evaluate` override) ⇒ operations fall back to the default and return a
# plain ParaMatrix. Proves the two-method interface is genuinely sufficient.
struct MinimalPara{T,S<:AbstractMatrix{T},C<:FunctionClass} <: AbstractParaMatrix{T,S,C}
    coeffs::Vector{S}
    class::C
end
ParaLinearAlgebra.coefficients(A::MinimalPara) = A.coeffs
ParaLinearAlgebra.function_class(A::MinimalPara) = A.class

function _tagged(d, cls; seed=0, tag="t")
    return TaggedPara(
        [randn(MersenneTwister(seed + 137i), ComplexF64, d, d) for i in 1:nbasis(cls)],
        cls,
        tag,
    )
end
# same coefficient data as `A`, but the canonical type — the oracle to compare against
_canon(A) = ParaMatrix(coefficients(A), function_class(A))

@testset "subtype recognized + minimal (2-method) interface evaluates" begin
    cls = Laurent(-2, 2)
    A = _tagged(3, cls; seed=3)
    @test A isa AbstractParaMatrix
    @test !(A isa ParaMatrix)
    @test eltype(A) == ComplexF64
    @test size(A) == (3, 3) && size(A, 1) == 3
    @test nterms(A) == nbasis(cls)
    @test coeff(A, 0) === coefficients(A)[3]                  # generic accessor on the subtype
    ref = _canon(A)
    M = MinimalPara(coefficients(A), function_class(A))       # only coefficients+function_class
    for θ in RNG_PTS
        @test A(θ) ≈ ref(θ)                                  # inherited evaluate (callable)
        @test evaluate(A, θ) ≈ ref(θ)
        @test M(θ) ≈ ref(θ)                                  # the 2-method subtype evaluates too
    end
    for θ in (0.13, 0.5, 0.81)                               # inherited ∂_p
        @test evaluate_deriv(A, θ) ≈ (A(θ + 1e-6) - A(θ - 1e-6)) / 2e-6 atol = 1e-5
    end
end

@testset "inherited ring / structural algebra (vs canonical ParaMatrix)" begin
    cls = Laurent(-1, 1)
    for d in (1, 2, 4), seed in (11, 47)
        A = _tagged(d, cls; seed=seed)
        B = _tagged(d, cls; seed=seed + 1)
        rA, rB = _canon(A), _canon(B)
        for θ in RNG_PTS
            @test (A + B)(θ) ≈ rA(θ) + rB(θ)
            @test (A - B)(θ) ≈ rA(θ) - rB(θ)
            @test (-A)(θ) ≈ -rA(θ)
            @test (2.5 * A)(θ) ≈ 2.5 * rA(θ)
            @test (A / 2)(θ) ≈ rA(θ) / 2
            @test (A * B)(θ) ≈ rA(θ) * rB(θ)                 # convolution ⟺ pointwise product
            @test (A ⊗ B)(θ) ≈ kron(rA(θ), rB(θ))
            @test (A ⊕ B)(θ) ≈ cat(rA(θ), rB(θ); dims=(1, 2))
            @test (A^3)(θ) ≈ rA(θ)^3
            @test transpose(A)(θ) ≈ transpose(rA(θ))
            @test tr(A)(θ)[1, 1] ≈ tr(rA(θ))
        end
        @test norm(A, 2) ≈ norm(rA, 2)                       # inherited L² norm
    end
end

@testset "_rebuild governs the return type (preserve vs fall back to ParaMatrix)" begin
    cls = Laurent(-1, 1)
    A = _tagged(2, cls; tag="keepme")
    B = _tagged(2, cls; seed=5, tag="other")
    # TaggedPara overrides _rebuild ⇒ every op stays TaggedPara and carries A's tag
    for op in (
        A + B,
        A - B,
        -A,
        A * B,
        2.0 * A,
        A / 2,
        A ⊗ B,
        A ⊕ B,
        A^2,
        para(A),
        transpose(A),
        copy(A),
        zero(A),
    )
        @test op isa TaggedPara
        @test op.tag == "keepme"
    end
    # MinimalPara has no _rebuild ⇒ the default builds a plain ParaMatrix
    M = MinimalPara(coefficients(A), function_class(A))
    @test (M + M) isa ParaMatrix
    @test (M * M) isa ParaMatrix
    @test para(M) isa ParaMatrix
    # binary ops dispatch on the FIRST argument's _rebuild (mixed concrete types OK)
    @test (A + _canon(B)) isa TaggedPara
    @test (_canon(A) + B) isa ParaMatrix
end

@testset "inherited Laurent para-structure + predicates" begin
    cls = Laurent(-2, 2)
    A = _tagged(3, cls; seed=23)
    rA = _canon(A)
    for θ in RNG_PTS
        @test para(A)(θ) ≈ adjoint(rA(θ))                    # on-circle para-adjoint
        @test A'(θ) ≈ adjoint(rA(θ))                         # `'` routes to para
        @test det(A)(θ)[1, 1] ≈ det(rA(θ))
    end
    H = parahermitianpart(A)                                 # built from the subtype
    @test H isa TaggedPara
    @test isparahermitian(H) && ishermitian(H)
    @test !isparaunitary(A)                                  # generic A is not para-unitary
    G = para(A) * A                                          # PSD Gram, TaggedPara
    @test ispositive(G) == ispositive(_canon(G))
    # a genuinely para-unitary subtype  D(θ) = diag(e^{2πiθ}, e^{-2πiθ}):  para(D)·D = I
    D = TaggedPara(
        [ComplexF64[0 0; 0 1], zeros(ComplexF64, 2, 2), ComplexF64[1 0; 0 0]],
        Laurent(-1, 1),
        "u",
    )
    @test isparaunitary(D) && isparaunitary(D')
end

@testset "inherited factorizations (eigvals / svdvals / spectral_factor / para_qr)" begin
    cls = Laurent(-1, 1)
    A = _tagged(3, cls; seed=71)
    rA = _canon(A)
    # sampled spectra (per-θ lists): identical machinery ⇒ same numbers as canonical
    @test specmatch(_sortc(reduce(vcat, eigvals(A))), _sortc(reduce(vcat, eigvals(rA))))
    @test sort(reduce(vcat, svdvals(A))) ≈ sort(reduce(vcat, svdvals(rA)))
    # exact spectral factor of a subtype-built PD para-Hermitian Gram: M·para(M) ≈ G
    M0 = _tagged(3, Analytic(1); seed=5)
    G = para(M0) * M0 + paraeye(3, ComplexF64, Laurent(-1, 1))
    @test G isa TaggedPara && ishermitian(G) && ispositive(G)
    W = spectral_factor(G; N=32)
    for θ in (0.1, 0.4, 0.7, 0.95)
        @test W(θ) * para(W)(θ) ≈ G(θ) atol = 1e-6
    end
    # parameterized → parameterized QR straight off the subtype
    F = para_qr(G; N=32, order=40)
    @test F.residual < 1e-6
    for θ in (0.1, 0.4, 0.7)
        @test G(θ) ≈ F.Q(θ) * F.R(θ) atol = 1e-6
    end
end
