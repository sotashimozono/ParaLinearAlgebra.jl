# classes/polynomial.jl — monomial class p^k (a real-parameter RING class).

"""
    Polynomial(N) <: FunctionClass

Monomial basis `p^k` for `k = 0:N` (`N+1` weights). A ring class (supports
`*`/`kron`) for a non-periodic real parameter `p`.
"""
struct Polynomial <: RingClass
    N::Int
    function Polynomial(N::Int)
        N ≥ 0 || throw(ArgumentError("Polynomial degree N must be ≥ 0; got $N"))
        return new(N)
    end
end

powers(c::Polynomial) = 0:(c.N)

basis(c::Polynomial, p) = [float(p)^k for k in powers(c)]
basis_deriv(c::Polynomial, p) =
    [k == 0 ? zero(float(p)) : k * float(p)^(k - 1) for k in powers(c)]

# L² Gram over [0,1):  ∫ θ^k θ^l dθ = 1/(k+l+1)  (a Hilbert-type matrix)
basis_gram(c::Polynomial) = [1.0 / (k + l + 1) for k in 0:(c.N), l in 0:(c.N)]

_prodclass(a::Polynomial, b::Polynomial) = Polynomial(a.N + b.N)
