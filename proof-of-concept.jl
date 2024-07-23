using MatrixDepot
using LinearAlgebra
using KLU
using SparseArrays
using Interpolations
using DataFrames
using Plots

# Wrappers to different LU algorithms and implementations
function umfpack_a(A)
    t = @elapsed  L, U, p = lu(A)
    err =  norm(A[p,:] - L*U, 1)
    return t, err
end

function klu_a(A)
    sA = @views sparse(A)
    t = @elapsed K = klu(sA)
    err = norm(K.L * K.U + K.F - K.Rs .\ A[K.p, K.q], 1)
    return t, err
end

# Define metrix names, sizes, and algorithms to explore

#mat_names = ["baart","binomial","blur","cauchy","chebspec","chow","circul","clement","companion","deriv2","dingdong","erdrey","fiedler","forsythe","foxgood","frank","gilbert","golub","gravity","grcar","hadamard","hankel","heat","hilb","invhilb","invol","kahan","kms","lehmer","lotkin","magic","minij","moler","neumann","oscillate","parallax","parter","pascal","pei","phillips","poisson","prolate","randcorr","rando","randsvd","rohess","rosser","sampling","shaw","smallworld","spikes","toeplitz","tridiag","triw","ursell","vand","wathen","wilkinson","wing"]
mat_names = ["baart"]
ns = [900]
algs = [umfpack_a, klu_a]

df = DataFrame(mat_name = String[], n = Int[], algorithm = String[], 
               t = Float64[], err = Float64[])
for mat_name in mat_names
    for n in ns
        A = matrixdepot(mat_name, n)
        for a in algs
            t, err = a(A)
            push!(df, [mat_name, n, "$(nameof(a))", t, err])
        end
    end
end

