using ParaLinearAlgebra
using LinearAlgebra
using Random
using SparseArrays
using StaticArrays
using Test, Aqua

include("helpers.jl")

# Test files mirror the src/ layout one-to-one, so it is explicit which source
# each test exercises:  test/<dir>/test_<name>.jl  ↔  src/<dir>/<name>.jl.
const TESTDIRS = ["core", "classes", "solver", "utils"]

@testset "ParaLinearAlgebra" begin
    @testset "Aqua" begin
        Aqua.test_all(ParaLinearAlgebra; ambiguities=false)
    end

    for dir in TESTDIRS
        dirpath = joinpath(@__DIR__, dir)
        files = sort(
            filter(f -> startswith(f, "test_") && endswith(f, ".jl"), readdir(dirpath))
        )
        isempty(files) && @warn "no test files in test/$dir"
        for f in files
            @testset "$dir/$f" begin
                include(joinpath(dirpath, f))
            end
        end
    end
end
