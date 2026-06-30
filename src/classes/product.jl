# classes/product.jl — product of classes for a multi-parameter object
#   A(p₁,…,pₙ) = Σ_{k⃗} C_{k⃗} ∏ᵢ basisᵢ(pᵢ)[kᵢ].
# Multi-powers are CartesianIndex so they add/negate/compare like scalar powers.

"""
    ProductClass(classes...) <: FunctionClass

The product of one-parameter classes, for a parameterized matrix depending on a
tuple of parameters. Ring structure (and para-adjoint, for all-`Laurent`
products) is inherited axis-wise.
"""
struct ProductClass{TT<:Tuple} <: RingClass
    classes::TT
end
ProductClass(cs::FunctionClass...) = ProductClass(cs)

function powers(pc::ProductClass)
    return vec([CartesianIndex(t) for t in Iterators.product(map(powers, pc.classes)...)])
end
nbasis(pc::ProductClass) = prod(map(nbasis, pc.classes))

function basis(pc::ProductClass, ps)
    bs = map(basis, pc.classes, ps)
    return vec([
        prod(bs[d][I[d]] for d in eachindex(bs)) for I in CartesianIndices(map(length, bs))
    ])
end

# partial derivative along axis `dim` (product rule): differentiate factor `dim`,
# keep the others. The scalar 2-arg form is undefined for a multi-parameter class.
function basis_deriv(pc::ProductClass, ps, dim::Integer)
    1 ≤ dim ≤ length(pc.classes) ||
        throw(ArgumentError("dim=$dim out of range 1:$(length(pc.classes))"))
    # keep `bs` a TUPLE so `map(length, bs)` is a tuple and CartesianIndices is N-D
    bs = ntuple(
        d -> d == dim ? basis_deriv(pc.classes[d], ps[d]) : basis(pc.classes[d], ps[d]),
        length(pc.classes),
    )
    return vec([
        prod(bs[d][I[d]] for d in eachindex(bs)) for I in CartesianIndices(map(length, bs))
    ])
end
function basis_deriv(pc::ProductClass, ps)
    return throw(
        ArgumentError(
            "ProductClass derivative is per-axis: call basis_deriv(class, ps, dim) or " *
            "evaluate_deriv(A, ps, dim), not the scalar 2-arg form",
        ),
    )
end

# separable L² Gram:  M_{IJ} = ∏_d M^d_{I[d],J[d]}
function basis_gram(pc::ProductClass)
    grams = map(basis_gram, pc.classes)
    loc = map(c -> Dict(p => i for (i, p) in enumerate(powers(c))), pc.classes)
    pw = powers(pc)
    n = length(pw)
    M = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        M[i, j] = prod(
            grams[d][loc[d][pw[i][d]], loc[d][pw[j][d]]] for d in eachindex(grams)
        )
    end
    return M
end

# separable integral:  (∫∏ wᵈ) = ∏ (∫ wᵈ)
function basis_integral(pc::ProductClass)
    vs = map(basis_integral, pc.classes)
    return vec([
        prod(vs[d][I[d]] for d in eachindex(vs)) for I in CartesianIndices(map(length, vs))
    ])
end

function _prodclass(a::ProductClass, b::ProductClass)
    return ProductClass(map(_prodclass, a.classes, b.classes))
end

# multi-parameter para-adjoint: negate every axis index, adjoint each block (all-Laurent)
function para(A::ParaMatrix{T,S,<:ProductClass}) where {T,S}
    pc = A.class
    all(c isa Laurent for c in pc.classes) ||
        error("para needs an all-Laurent product class")
    npc = ProductClass(map(c -> Laurent(-c.hi, -c.lo), pc.classes))
    op = powers(pc)
    idx = Dict(op[i] => i for i in eachindex(op))
    return ParaMatrix([_adj(A.coeffs[idx[-m]]) for m in powers(npc)], npc)
end
