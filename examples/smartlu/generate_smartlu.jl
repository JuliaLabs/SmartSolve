# SmartSolve aims to significantly accelerate various linear
# algebra algorithms based on providing better algorithmic
# and architectural choices.
# Here, SmartSolve is used to automatically generate an
# optimized version of LU decomposition: SmartLU.

cd(@__DIR__)
using Pkg
Pkg.activate(".")

using SmartSolve
using LinearAlgebra
using SparseArrays
using KLU
using SuperLU
using MatrixDepot
using BenchmarkTools
using DecisionTree
using BSON

import SmartSolve: compute_feature_values

# OpenBLAS vs MKL
mkl = true
if mkl
    using MKL
end
BLAS.get_config()

# Define candidate algorithms
dgetrf(A::Matrix) = lu(A)
dgetrf(A::SparseMatrixCSC) = lu(Matrix(A))
umfpack(A::Matrix) = lu(sparse(A))
umfpack(A::SparseMatrixCSC) = lu(A)
KLU.klu(A::Matrix) = klu(sparse(A))
SuperLU.splu(A::Matrix) = splu(sparse(A))
algs = [dgetrf, umfpack, klu, splu]

# Define your custom matrices to be included in training
n = 2^12;
A = rand(n, n)
B = sprand(n, n, 0.3)
mats = [A, B]

# Generate a smart version of your algorithm
alg_name  = "lu"
alg_path = "smart$alg_name/"
smartsolve(alg_path, alg_name, algs;
           mats = mats, ns = [2^4, 2^8, 2^12])

# Include the newly generated algorithm
include("$alg_path/smart$alg_name.jl")

# Benchmark classical vs smart algorithm
n = 2^12;
benchmark_seconds = 2 # 200
A = matrixdepot("poisson", round(Int, sqrt(n))); # nxn
@benchmark lu($A) seconds=benchmark_seconds
@benchmark smartlu($A) seconds=benchmark_seconds

# Benchmark Backslash vs SmartBackslash (via SmartLU)
b = rand(n);
@benchmark $A\$b seconds=benchmark_seconds
@benchmark lu($A)\$b seconds=benchmark_seconds
@benchmark smartlu($A)\$b seconds=benchmark_seconds

# Compute errors
x = A \ b;
norm(A * x - b, 1)
x = lu(A) \ b;
norm(A * x - b, 1)
x = smartlu(A) \ b;
norm(A * x - b, 1)

# Plot results
makeplots(alg_path, alg_name)

