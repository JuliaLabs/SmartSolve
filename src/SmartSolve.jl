module SmartSolve

using MatrixDepot
using LinearAlgebra
using DataFrames
using OrderedCollections
using CSV
using PlotlyJS
using DecisionTree
using Random
using BenchmarkTools
using BSON

using ShapML
using MLJ
using Plots
using TreeRecipe
using CategoricalArrays
using SparseArrays
using ColorSchemes

include("SmartDiscovery.jl")
include("SmartDB.jl")
include("SmartModel.jl")
include("Utils.jl")

end # module SmartSolve
