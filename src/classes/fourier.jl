# classes/fourier.jl — real truncated Fourier series (an ANSATZ class).
#
# A self-contained plug-in: it implements only basis/basis_deriv/nbasis. It is NOT
# a ring class (no `powers`/`_prodclass`), so it supports construct + evaluate +
# ∂_p + AD, but not the ParaMatrix product/para-adjoint (use `Laurent` for those).

"""
    Fourier(K) <: FunctionClass

Real truncated Fourier series in the angle `θ ∈ R/Z`, ordered
`[1, cos2πkθ (k=1..K), sin2πkθ (k=1..K)]` (`2K+1` real weights). Suited to
real-valued ansätze; differentiable in `θ`.
"""
struct Fourier <: FunctionClass
    K::Int
    function Fourier(K::Int)
        K ≥ 0 || throw(ArgumentError("Fourier order K must be ≥ 0; got $K"))
        return new(K)
    end
end

nbasis(c::Fourier) = 2c.K + 1

# `cospi`/`sinpi` evaluate cos(πx)/sin(πx) without the 2π argument-reduction
# rounding error, so accuracy holds at the Float64 ulp and extends to BigFloat.
function basis(c::Fourier, p)
    T = typeof(float(p))
    K = c.K
    w = Vector{T}(undef, 2K + 1)
    w[1] = one(T)
    @inbounds for k in 1:K
        w[1 + k] = cospi(2 * k * p)
        w[1 + K + k] = sinpi(2 * k * p)
    end
    return w
end

function basis_deriv(c::Fourier, p)
    T = typeof(float(p))
    K = c.K
    τ = oftype(zero(T), π)          # full-precision π in p's type (BigFloat-safe)
    w = zeros(T, 2K + 1)
    @inbounds for k in 1:K
        w[1 + k] = -2 * k * τ * sinpi(2 * k * p)
        w[1 + K + k] = 2 * k * τ * cospi(2 * k * p)
    end
    return w
end

# L² Gram over [0,1):  ∫1²=1, ∫cos²(2πkθ)=∫sin²(2πkθ)=½, all cross terms 0.
basis_gram(c::Fourier) = Matrix(Diagonal([1.0; fill(0.5, 2c.K)]))

# ∫₀¹: only the constant term survives (∫cos = ∫sin = 0)
basis_integral(c::Fourier) = [k == 1 ? 1.0 : 0.0 for k in 1:(2c.K + 1)]
