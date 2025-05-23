# WIP

# SmartSolve aims to significantly accelerate various linear
# algebra algorithms based on providing better algorithmic
# and architectural choices.
# Here, SmartSolve is used to automatically generate an
# optimized version of QR decomposition: SmartQR.

cd(@__DIR__)
using Pkg
Pkg.activate(".")

using SmartSolve
using LinearAlgebra
using SparseArrays
using BandedMatrices
using MatrixDepot
using BenchmarkTools
using DecisionTree
using BSON
# TODO: add missing dependencies

import SmartSolve: compute_feature_values

# OpenBLAS vs MKL
mkl = false
if mkl
    using MKL
end
BLAS.get_config()

# Define candidate algorithms

# TODO: add missing algorithms
#dgetrf(A::Matrix) = lu(A)
#dgetrf(A::SparseMatrixCSC) = lu(Matrix(A))
#dgetrf(A::SparseMatrixCSC{Bool, Int64}) = lu(Matrix(A))
#dgetrf(A::Symmetric) = lu(A.data)
#bandedlu(A::Matrix) = BandedMatrices.lu(BandedMatrix(sparse(A)))
#bandedlu(A::SparseMatrixCSC{Float64, Int64}) = BandedMatrices.lu(BandedMatrix(A))
#bandedlu(A::SparseMatrixCSC{Int64, Int64}) = BandedMatrices.lu(BandedMatrix(Float64.(A)))
#bandedlu(A::SparseMatrixCSC{Bool, Int64}) = BandedMatrices.lu(BandedMatrix(Float64.(A)))
#bandedlu(A::Symmetric) = BandedMatrices.lu(Float64.(BandedMatrix(sparse(A.data))))
#algs = [dgetrf, bandedlu]

# Define your custom matrices to be included in training (WIP)
n = 2^12;
A = rand(n, n)
B = sprand(n, n, 0.3)
mats = [A, B]

# Generate a smart version of your algorithm
alg_name  = "qr"
alg_path = "smart$alg_name/"
smartsolve(alg_path, alg_name, algs; n_experiments = 1,
           mats = mats, ns = [2^8], features = [:isbandedpattern])

# Include the newly generated algorithm
include("$alg_path/smart$alg_name.jl")

# Benchmark classical vs smart algorithm
n = 2^12;
benchmark_seconds = 2 # 200
A = Matrix(matrixdepot("poisson", round(Int, sqrt(n)))); # nxn
println("** Benchmark Time for Regular QR Decomposition")
display(@benchmark qr($A) seconds=benchmark_seconds)
println("** Benchmark Time for Smart QR Decomposition")
display(@benchmark smartqr($A) seconds=benchmark_seconds)

# Benchmark Backslash vs SmartBackslash (via SmartQR)
b = rand(n);
println("** Benchmark Time for Backslash")
display(@benchmark $A\$b seconds=benchmark_seconds)
println("** Benchmark Time for Qr Backslash")
display(@benchmark qr($A)\$b seconds=benchmark_seconds)
println("** Benchmark Time for SmartBackslash")
display(@benchmark smartqr($A)\$b seconds=benchmark_seconds)

# Compute errors
x = A \ b;
norm(A * x - b, 1)
x = qr(A) \ b;
norm(A * x - b, 1)
x = smartqr(A) \ b;
norm(A * x - b, 1)

# Plot results
makeplots(alg_path, alg_name, tick_label_size=30, label_size=36, title_size=48)

