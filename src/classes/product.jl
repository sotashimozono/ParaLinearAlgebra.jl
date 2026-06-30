# classes/product.jl — product of classes for a multi-parameter object
#   A(p₁,…,pₙ) = Σ_{k⃗} C_{k⃗} ∏ᵢ basisᵢ(pᵢ)[kᵢ].
# Multi-powers are CartesianIndex so they add/negate/compare like scalar powers.

"""
    ProductClass(classes...) <: FunctionClass

The product of one-parameter classes, for a parameterized matrix depending on a
tuple of parameters. Ring structure (and para-adjoint, for all-`Laurent`
products) is inherited axis-wise.
"""
struct ProductClass{TT<:Tuple} <: FunctionClass
    classes::TT
end
ProductClass(cs::FunctionClass...) = ProductClass(cs)

powers(pc::ProductClass) =
    vec([CartesianIndex(t) for t in Iterators.product(map(powers, pc.classes)...)])
nbasis(pc::ProductClass) = prod(map(nbasis, pc.classes))

function basis(pc::ProductClass, ps)
    bs = map(basis, pc.classes, ps)
    return vec([
        prod(bs[d][I[d]] for d in eachindex(bs)) for I in CartesianIndices(map(length, bs))
    ])
end

_prodclass(a::ProductClass, b::ProductClass) =
    ProductClass(map(_prodclass, a.classes, b.classes))

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
