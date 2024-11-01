module SmartSolve

using MatrixDepot
using LinearAlgebra
using DataFrames
using OrderedCollections
using CSV
using CairoMakie
using DecisionTree
using Random
using BenchmarkTools
using BSON
using SparseArrays

include("SmartDiscovery.jl")
include("SmartDB.jl")
include("SmartModel.jl")
include("Utils.jl")

end # module SmartSolve
