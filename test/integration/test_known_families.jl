# Oracle tests across the whole library: well-known ONE-PARAMETER MATRIX GROUPS
# D(θ) (homomorphisms θ ↦ exp(θX)) satisfy closed-form identities
#     D(θ)⁻¹ = D(-θ),   D(θ)·D(φ) = D(θ+φ),   det D known,   spectrum known.
# A faithful library must reproduce these from its own `inv`/`para`/`*`/`det`/
# `eigvals` — non-tautological because the expected answers are pure group theory.

# `specmatch` (order-independent spectrum compare) lives in test/helpers.jl.

# check the group laws for a one-parameter para-unitary group D with known D(θ) and det
function check_group(D, Dknown, detknown)
    @test isparaunitary(D)                                          # D(θ) unitary ∀θ
    for θ in RNG_PTS
        @test D(θ) ≈ Dknown(θ) atol = 1e-10                         # construction
        @test inv(D)(θ) ≈ Dknown(-θ) atol = 1e-9                    # D⁻¹(θ) = D(-θ)  ← the property
        @test para(D)(θ) ≈ Dknown(-θ) atol = 1e-9                   # para-adjoint = D(-θ)
        @test (D * D)(θ) ≈ Dknown(2θ) atol = 1e-9                   # D·D = D(2θ)  (homomorphism)
        @test (D * inv(D))(θ) ≈ I atol = 1e-9                       # D·D⁻¹ = I
        @test det(D)(θ)[1, 1] ≈ detknown(θ) atol = 1e-8            # determinant closed form
    end
end

@testset "rotation group SO(2): R(θ)⁻¹ = R(-θ)" begin
    Id = ComplexF64[1 0; 0 1]
    J = ComplexF64[0 -1; 1 0]
    R = ParaMatrix(
        [0.5Id + 0.5im * J, zeros(ComplexF64, 2, 2), 0.5Id - 0.5im * J], Laurent(-1, 1)
    )
    Rknown(θ) = [cospi(2θ) -sinpi(2θ); sinpi(2θ) cospi(2θ)]
    check_group(R, Rknown, θ -> 1.0)                                # det R = 1
    for θ in RNG_PTS
        @test specmatch(eigvals(R(θ)), [cispi(2θ), cispi(-2θ)])  # spectrum e^{±2πiθ}
    end
end

@testset "SU(2) subgroup exp(2πiθ σx): U(θ)⁻¹ = U(-θ)" begin
    Id = ComplexF64[1 0; 0 1]
    sx = ComplexF64[0 1; 1 0]
    U = ParaMatrix([0.5Id - 0.5sx, zeros(ComplexF64, 2, 2), 0.5Id + 0.5sx], Laurent(-1, 1))
    Uknown(θ) = [cospi(2θ) im*sinpi(2θ); im*sinpi(2θ) cospi(2θ)]
    check_group(U, Uknown, θ -> 1.0)                                # det U = 1 (SU(2))
end

@testset "diagonal character D(θ)=diag(z^k): D(θ)⁻¹ = D(-θ)" begin
    ks = [-1, 0, 2]
    lo, hi = minimum(ks), maximum(ks)
    coeffs = [Diagonal(ComplexF64[k == p ? 1 : 0 for k in ks]) for p in lo:hi]
    D = ParaMatrix(coeffs, Laurent(lo, hi))
    Dknown(θ) = Diagonal([cispi(2 * k * θ) for k in ks])
    check_group(D, Dknown, θ -> cispi(2 * sum(ks) * θ))             # det = z^{Σk}
    for θ in RNG_PTS
        @test specmatch(eigvals(Matrix(D(θ))), [cispi(2 * k * θ) for k in ks])
    end
end

# The Laurent families above all live in one ring class; the POLYNOMIAL ring class
# gets its own closed-form oracle from the Chebyshev product-to-sum identity
#     Tₘ(x)·Tₙ(x) = ½( T_{m+n}(x) + T_{|m−n|}(x) ),
# with the independent defining property Tₙ(cos φ) = cos(n φ). This pins down the
# Polynomial-ring product `*` (_convolve) against pure trigonometry.
@testset "Chebyshev product-to-sum (Polynomial ring): Tₘ·Tₙ = ½(T_{m+n}+T_{|m−n|})" begin
    # monomial coefficients of Tₙ via the recurrence T_{k+1} = 2x·T_k − T_{k−1}
    function chebcoeffs(n)
        n == 0 && return [1.0]
        n == 1 && return [0.0, 1.0]
        tkm1, tk = [1.0], [0.0, 1.0]
        for _ in 2:n
            t = vcat(0.0, 2 .* tk)              # 2x · T_k
            for i in eachindex(tkm1)
                t[i] -= tkm1[i]                 # − T_{k−1}
            end
            tkm1, tk = tk, t
        end
        return tk
    end
    chebpoly(n) = ParaMatrix([fill(c, 1, 1) for c in chebcoeffs(n)], Polynomial(n))
    Tval(n, t) = cospi(n * t)                   # Tₙ(cos πt) = cos(nπt) — definition, not construction

    for n in 0:4, t in RNG_PTS
        @test chebpoly(n)(cospi(t))[1, 1] ≈ Tval(n, t) atol = 1e-10   # construction matches definition
    end
    for (m, n) in ((1, 4), (2, 3), (2, 2), (3, 4), (0, 3))
        P = chebpoly(m) * chebpoly(n)           # the Polynomial-ring product under test
        @test P.class == Polynomial(m + n)      # degrees add
        for t in RNG_PTS
            x = cospi(t)
            @test P(x)[1, 1] ≈ 0.5 * (Tval(m + n, t) + Tval(abs(m - n), t)) atol = 1e-10  # product-to-sum
            @test P(x)[1, 1] ≈ chebpoly(m)(x)[1, 1] * chebpoly(n)(x)[1, 1] atol = 1e-10   # vs pointwise
        end
    end
end
