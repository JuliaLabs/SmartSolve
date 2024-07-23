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

# All matrices
#mat_names =  ["baart","binomial","blur","cauchy","chebspec","chow","circul",
#              "clement","companion","deriv2","dingdong","erdrey","fiedler",
#              "forsythe","foxgood","frank","gilbert","golub","gravity","grcar",
#              "hadamard","hankel","heat","hilb","invhilb","invol","kahan","kms",
#              "lehmer","lotkin","magic","minij","moler","neumann","oscillate",
#              "parallax","parter","pascal","pei","phillips","poisson","prolate",
#              "randcorr","rando","randsvd","rohess","rosser","sampling","shaw",
#              "smallworld","spikes","toeplitz","tridiag","triw","ursell","vand",
#              "wathen","wilkinson","wing"]

# Problematic matrices: interface issues (e.g. dim^2 instead of dim), out of memory, singular value exception.
# mat_names = ["binomial", "blur", "poisson", "chow", "erdrey", "invol","neumann",
#              "parallax","pascal","rosser",,"vand", "smallworld","gilbert",
#              "hadamard","heat","invhilb", "wathen"] 

# Matrices
mat_names = ["baart", "cauchy", "chebspec", "circul", "clement", "companion", 
             "deriv2", "dingdong", "fiedler",
             "forsythe", "foxgood","frank","golub","gravity","grcar",
             "hankel","hilb","kahan","kms",
             "lehmer", "lotkin","magic","minij","moler","oscillate",
             "parter","pei","phillips","prolate",
             "randcorr","rando","randsvd","rohess","sampling","shaw",
             "spikes","toeplitz","tridiag","triw","ursell",
             "wilkinson","wing"]
# Matrix sizes
ns = [900]

# Algorithms
algs = [umfpack_a, klu_a]

# Evaluate
df = DataFrame(mat_name = String[], n = Int[], algorithm = String[], 
               t = Float64[], err = Float64[])
for mat_name in mat_names
    for n in ns
        A = matrixdepot(mat_name, n)
        for a in algs
            try
                t, err = a(A)
                push!(df, [mat_name, n, "$(nameof(a))", t, err])
            catch e
                println("$(mat_name),$n, $(nameof(a)): an error occurred: $e")
            end
        end
        GC.gc()
    end
end

df

