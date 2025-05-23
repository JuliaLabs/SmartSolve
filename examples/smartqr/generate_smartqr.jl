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
using SparseMatricesCSR: SparseMatrixCSR
using BandedMatrices
using MatrixDepot
using BenchmarkTools
using DecisionTree
using BSON

import SmartSolve: compute_feature_values

# OpenBLAS vs MKL
mkl = false
if mkl
    using MKL
end
BLAS.get_config()

# Define candidate algorithms

geqrt(A::Matrix) = LAPACK.geqrt!(A)[1]
geqrt(A::SparseMatrixCSC) = LAPACK.geqrt!(Matrix(A), size(A,2)÷2)[1]
geqrt(A::SparseMatrixCSC{Bool, Int64}) = LAPACK.geqrt!(Matrix(A), size(A,2)÷2)[1]
geqrt(A::SparseMatrixCSR) = LAPACK.geqrt!(Matrix(A), size(A,2)÷2)[1]
geqrt(A::SparseMatrixCSR{Bool, Int64}) = LAPACK.geqrt!(Matrix(A), size(A,2)÷2)[1]
geqrt(A::Symmetric) = LAPACK.geqrt!(A.data, size(A.data,2)÷2)[1]

geqrt3(A::Matrix) = LAPACK.geqrt3!(A)[1]
geqrt3(A::SparseMatrixCSC) = LAPACK.geqrt3!(Matrix(A))[1]
geqrt3(A::SparseMatrixCSC{Bool, Int64}) = LAPACK.geqrt3!(Matrix(A))[1]
geqrt3(A::SparseMatrixCSR) = LAPACK.geqrt3!(Matrix(A))[1]
geqrt3(A::SparseMatrixCSR{Bool, Int64}) = LAPACK.geqrt3!(Matrix(A))[1]
geqrt3(A::Symmetric) = LAPACK.geqrt3!(A.data)[1]

geqrf(A::Matrix) = LAPACK.geqrf!(A)[1]
geqrf(A::SparseMatrixCSC) = LAPACK.geqrf!(Matrix(A))[1]
geqrf(A::SparseMatrixCSC{Bool, Int64}) = LAPACK.geqrf!(Matrix(A))[1]
geqrf(A::SparseMatrixCSR) = LAPACK.geqrf!(Matrix(A))[1]
geqrf(A::SparseMatrixCSR{Bool, Int64}) = LAPACK.geqrf!(Matrix(A))[1]
geqrf(A::Symmetric) = LAPACK.geqrf!(A.data)[1]

bandedqr(A::Matrix) = BandedMatrices.banded_qr!(BandedMatrix(sparse(A)))
bandedqr(A::SparseMatrixCSC{Float64, Int64}) = BandedMatrices.banded_qr!(BandedMatrix(A))
bandedqr(A::SparseMatrixCSC{Int64, Int64}) = BandedMatrices.banded_qr!(BandedMatrix(Float64.(A)))
bandedqr(A::SparseMatrixCSC{Bool, Int64}) = BandedMatrices.banded_qr!(BandedMatrix(Float64.(A)))
bandedqr(A::SparseMatrixCSR{Float64, Int64}) = BandedMatrices.banded_qr!(BandedMatrix(A))
bandedqr(A::SparseMatrixCSR{Int64, Int64}) = BandedMatrices.banded_qr!(BandedMatrix(Float64.(A)))
bandedqr(A::SparseMatrixCSR{Bool, Int64}) = BandedMatrices.banded_qr!(BandedMatrix(Float64.(A)))
bandedqr(A::Symmetric) = BandedMatrices.banded_qr!(Float64.(BandedMatrix(sparse(A.data))))

algs = [geqrt, geqrt3, geqrf, bandedqr]

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
#           mats = mats, ns = [2^8], features = [:isbandedpattern], castings = [Matrix, SparseMatrixCSC])

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

