using MatrixDepot
using LinearAlgebra
using KLU
using SuperLU
using SparseArrays
using Interpolations
using DataFrames
using OrderedCollections
using CSV
using PlotlyJS
using DecisionTree
using Random
using BenchmarkTools
using BSON

# OpenBLAS vs MKL
mkl = true
if mkl
    using MKL
end
BLAS.get_config()

include("SmartSolve.jl")

# Use SmartSolve to generate a smart version of a
# linear algebra algorithm based on improved algorithm choices

# Case study: SmartLU

# Define candidate algorithms
dgetrf(A::Matrix) = lu(A)
dgetrf(A::SparseMatrixCSC) = lu(Matrix(A))
umfpack(A::Matrix) = lu(sparse(A))
umfpack(A::SparseMatrixCSC) = lu(A)
KLU.klu(A::Matrix) = klu(sparse(A))
SuperLU.splu(A::Matrix) = splu(sparse(A))
algs = [dgetrf, umfpack, klu, splu]

# Define your custom matrices to be included in training
n = 2^14;
A = rand(n, n)
B = sprand(n, n, 0.3)
mats = [A, B]

# Generate the smart version of your algorithm
alg_name  = "lu"
alg_path = "smartalgs/$alg_name"
smartsolve(alg_path, alg_name, algs; mats = mats)

# Include the newly generated algorithm
include("$alg_path/smart$alg_name.jl")

# Benchmark classical vs smart algorithm
n = 2^14;
A = matrixdepot("poisson", round(Int, sqrt(n))); # nxn
@benchmark lu($A) seconds=20
@benchmark smartlu($A) seconds=10

# Benchmark Backslash vs SmartBackslash (via SmartLU)
b = rand(n);
@benchmark $A\$b seconds=200
@benchmark lu($A)\$b seconds=200
@benchmark smartlu($A)\$b seconds=10

# Compute errors
x = A \ b;
norm(A * x - b, 1)
x = lu(A) \ b;
norm(A * x - b, 1)
x = smartlu(A) \ b;
norm(A * x - b, 1)

# Plot results
smartdb = CSV.read("$alg_path/smartdb-$alg_name.csv", DataFrame)
fulldb = CSV.read("$alg_path/fulldb-$alg_name.csv", DataFrame)
algs = BSON.load("$alg_path/algs-$alg_name.bson")[:algs]
ns = unique(smartdb[:, :n_cols])
for alg in algs
    local alg_name = String(Symbol(alg)) 
    alg_patterns = unique(smartdb[smartdb.algorithm .== alg_name, :pattern])
    plot_benchmark(alg_path, alg_name, fulldb, ns, algs, alg_patterns, "log")
end


#n = 2^12
#A = sprand(n, n, 0.5)
#B = Matrix(A)
#C = rand(n, n)
# @code_typed lu(A) # umfpack
# @code_typed lu(Matrix(A)) # getrf
# @code_typed lu(B) #  getrf
# @code_typed lu(C) # getrf
# @code_typed lu(Matrix(C)) #Compute getrf
# @benchmark lu($B) seconds=10 # 206.159ms
# @benchmark lu(Matrix($A)) seconds=10 # 314.668ms
# @benchmark lu($C) seconds=10 # 217.197 ms
# @benchmark lu(Matrix($C)) seconds=10 #  314.556 ms

# lu(sparse(A)) # umfpack
# lu(sparse(A)) # umfpack
# lu(sparse(B)) # umfpack
# lu(sparse(C)) # umfpack
# @benchmark lu($A) seconds=20 # umfpack, 2.123 s
# @benchmark lu(sparse($A)) seconds=20 # umfpack, 2.346 s s
# @benchmark klu($A) seconds=20 # klu, 26.511 s
# @benchmark klu(sparse($A)) seconds=20 # klu, 26.561s
# @benchmark splu($A) seconds=20 # splu, 3.840 s
# @benchmark splu(sparse($A)) seconds=20 # splu, 3.889 s
