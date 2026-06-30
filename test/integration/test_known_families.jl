# Oracle tests across the whole library: well-known ONE-PARAMETER MATRIX GROUPS
# D(θ) (homomorphisms θ ↦ exp(θX)) satisfy closed-form identities
#     D(θ)⁻¹ = D(-θ),   D(θ)·D(φ) = D(θ+φ),   det D known,   spectrum known.
# A faithful library must reproduce these from its own `inv`/`para`/`*`/`det`/
# `eigvals` — non-tautological because the expected answers are pure group theory.

# order-independent spectrum match: greedily pair each computed eigenvalue with the
# nearest expected one (robust to LAPACK ulp-noise that makes sort tie-breaks flip,
# e.g. for a conjugate pair a±bi whose real parts are equal by construction).
function specmatch(computed, expected; atol=1e-8)
    length(computed) == length(expected) || return false
    pool = collect(ComplexF64, expected)
    for x in computed
        i = argmin(abs.(pool .- x))
        abs(pool[i] - x) ≤ atol || return false
        deleteat!(pool, i)
    end
    return true
end

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
