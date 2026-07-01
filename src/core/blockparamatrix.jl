# core/blockparamatrix.jl — a block matrix whose blocks are INDEPENDENTLY
# parameterized. Each block is either a `ParaMatrix` (with its OWN class and its
# OWN parameter(s)) or a plain constant `AbstractMatrix`; the whole object depends
# on the CONCATENATION of the blocks' parameters, assigned in row-major order. This
# is the heterogeneous complement to `ParaMatrix` (one class, shared parameters):
# different blocks may use different classes and different parameters, e.g.
#     M(a, b) = [A(a)  0 ; 0  B(b)]        # A, B of possibly different classes.
# A `ParaMatrix` over a `ProductClass` is still the right tool when every entry
# shares the same parameters; `BlockParaMatrix` is for genuinely independent ones.

function _blockarity(B::AbstractParaMatrix)
    return function_class(B) isa ProductClass ? length(function_class(B).classes) : 1
end
_blockarity(::AbstractMatrix) = 0

"""
    BlockParaMatrix(rows)

A block matrix of independently-parameterized blocks. `rows` is a vector of block
rows (each a vector of blocks), or a `Matrix` of blocks; every block is a
[`ParaMatrix`](@ref) or a constant `AbstractMatrix`. Block sizes must be
consistent (common height per row, common width per column).

The object depends on the concatenation of the blocks' parameters (row-major):
each `ParaMatrix` block consumes as many global parameters as its class arity (1
for a one-parameter class, `n` for an `n`-axis `ProductClass`), constants consume
none. It is callable — `M(p)`, with `p` an `nparams`-tuple — assembling the dense
block matrix, each block evaluated at its own slice of `p`. Bands/parameters of
different blocks are independent: `M(a,b) = [A(a); B(b)]`.
"""
struct BlockParaMatrix{T}
    blocks::Matrix{Any}
    slices::Matrix{UnitRange{Int}}
    nparams::Int
    rowsz::Vector{Int}
    colsz::Vector{Int}
end

function BlockParaMatrix(blocks::AbstractMatrix)
    nr, nc = size(blocks)
    (nr > 0 && nc > 0) || throw(ArgumentError("BlockParaMatrix needs at least one block"))
    slices = Matrix{UnitRange{Int}}(undef, nr, nc)
    k = 0
    for i in 1:nr, j in 1:nc                         # row-major parameter assignment
        a = _blockarity(blocks[i, j])
        slices[i, j] = (k + 1):(k + a)
        k += a
    end
    rowsz = [size(blocks[i, 1], 1) for i in 1:nr]
    colsz = [size(blocks[1, j], 2) for j in 1:nc]
    for i in 1:nr, j in 1:nc
        size(blocks[i, j], 1) == rowsz[i] ||
            throw(DimensionMismatch("block row $i has inconsistent heights"))
        size(blocks[i, j], 2) == colsz[j] ||
            throw(DimensionMismatch("block column $j has inconsistent widths"))
    end
    T = promote_type((eltype(blocks[i, j]) for i in 1:nr for j in 1:nc)...)
    return BlockParaMatrix{T}(Matrix{Any}(blocks), slices, k, rowsz, colsz)
end

# vector-of-rows convenience: BlockParaMatrix([[A, Z], [Z, B]])
function BlockParaMatrix(rows::AbstractVector{<:AbstractVector})
    nc = length(first(rows))
    all(length(r) == nc for r in rows) ||
        throw(ArgumentError("all block rows must have the same number of blocks"))
    M = Matrix{Any}(undef, length(rows), nc)
    for (i, r) in enumerate(rows), j in 1:nc
        M[i, j] = r[j]
    end
    return BlockParaMatrix(M)
end

Base.size(M::BlockParaMatrix) = (sum(M.rowsz), sum(M.colsz))
Base.size(M::BlockParaMatrix, d::Int) = size(M)[d]
Base.eltype(::BlockParaMatrix{T}) where {T} = T

"""
    nparams(M::BlockParaMatrix) -> Int

The number of (independent) parameters `M` consumes — the sum of the blocks' arities.
"""
nparams(M::BlockParaMatrix) = M.nparams

function _evalblock(B::AbstractParaMatrix, sl, p)
    return length(sl) == 1 ? B(p[first(sl)]) : B(ntuple(k -> p[sl[k]], length(sl)))
end
_evalblock(B::AbstractMatrix, _, _) = B

function (M::BlockParaMatrix)(p)
    length(p) == M.nparams || throw(
        ArgumentError("BlockParaMatrix takes $(M.nparams) parameters; got $(length(p))")
    )
    nr, nc = size(M.blocks)
    rowmats = map(1:nr) do i
        return reduce(hcat, (_evalblock(M.blocks[i, j], M.slices[i, j], p) for j in 1:nc))
    end
    return reduce(vcat, rowmats)
end
