using MatrixDepot
using LinearAlgebra
using KLU
using SparseArrays
using Interpolations
using DataFrames
using CSV
using PlotlyJS

# Smart discovery ##############################################################

# Define wrappers to different LU algorithms and implementations
#function umfpack_a(A)
#    t = @elapsed  L, U, p = lu(A)
#    err = norm(A[p,:] - L*U, 1)
#    return t, err
#end
#function klu_a(A)
#    sA = @views sparse(A)
#    t = @elapsed K = klu(sA)
#    err = norm(K.L * K.U + K.F - K.Rs .\ A[K.p, K.q], 1)
#    return t, err
#end
function umfpack_a(A)
    t = @elapsed lu(A)
    return t, 0.0
end
function klu_a(A)
    t = @elapsed klu(sparse(A))
    return t, 0.0
end
algs = [umfpack_a, klu_a]

# Define matrices
mat_names = ["rosser", "companion", "forsythe", "grcar", "triw", "blur", "poisson",
             "heat", "baart", "cauchy", "circul", "clement", "deriv2",
             "dingdong", "fiedler",  "foxgood","frank", "golub","gravity",
             "hankel","hilb","kahan","kms", "lehmer", "lotkin","magic", "minij",
             "moler","oscillate", "parter", "pei", "prolate", "randcorr",
             "rando","randsvd","rohess", "sampling","shaw", "spikes", "toeplitz",
             "tridiag","ursell", "wilkinson","wing", "hadamard", "phillips"]

# Define matrix sizes
ns = [2^6, 2^8, 2^10, 2^12]

# Define number of experiments
n_experiments = 5

# Generate smart choice database through smart discovery
df = DataFrame(mat_name = String[], n = Int[], algorithm = String[], 
               time = Float64[], error = Float64[])
for i in 1:n_experiments
    println("Experiment $i")
    for mat_name in mat_names
        for n in ns
            # Generate matrix
            if mat_name in ["blur", "poisson"]
                n′ = convert(Int, sqrt(n))
            else
                n′ = n
            end
            A = matrixdepot(mat_name, n′)
            if size(A) != (n, n)
                println("Check matrix size: $(mat_name), ($n, $n) vs $(size(A))")
            end
            # Evaluate different algorithms
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
end


# Show and save discovery process results ######################################

df
CSV.write("smartsolve.csv", df)

function plot_benchmark(df, ns, algs, mat_names, xaxis_type)
    algs_str = ["$(nameof(a))" for a in algs]
    for n in ns
        p = plot(
            [(
                ts = [
                       (
                           df′ = @views df[(df.mat_name .== mat_name) .&&
                                           (df.n .== n) .&& 
                                           (df.algorithm .== a), :];
                           if length(df′.time) > 0
                              minimum(df′.time)
                           else
                              0.0
                           end
                       )
                       for mat_name in reverse(mat_names)
                     ];
                 bar(name=a, x=ts, y=reverse(mat_names), orientation="h")
                ) for a in algs_str
             ])
        relayout!(p, barmode="group",
                     xaxis_type=xaxis_type,
                     xaxis_title="Time [s]",
                     yaxis_title="Matrix name, size $(n)x$(n)")
        savefig(p, "algorithms_times_$(n)_$(xaxis_type).png", width=600, height=800, scale=1.5)
    end
end

plot_benchmark(df, ns, algs, mat_names, "log")
plot_benchmark(df, ns, algs, mat_names[1:8], "linear")


# Generate smart choice model ##################################################

#TODO: ML model here

smart_choice = Dict()
for mat_name in mat_names
    for n in ns
        df′ = @views df[(df.mat_name .== mat_name) .&& (df.n .== n), :]
        min_time = minimum(df′.time)
        min_time_row = df′[df′.time .== min_time, :]
        push!(smart_choice, (mat_name, n) => eval(Meta.parse(min_time_row.algorithm[1])))
    end
end

############

# All matrices
# mat_names =  ["baart","binomial","blur","cauchy","chebspec","chow","circul",
#              "clement","companion","deriv2","dingdong","erdrey","fiedler",
#              "forsythe","foxgood","frank","gilbert","golub","gravity","grcar",
#              "hadamard","hankel","heat","hilb","invhilb","invol","kahan","kms",
#              "lehmer","lotkin","magic","minij","moler","neumann","oscillate",
#              "parallax","parter","pascal","pei","phillips","poisson","prolate",
#              "randcorr","rando","randsvd","rohess","rosser","sampling","shaw",
#              "smallworld","spikes","toeplitz","tridiag","triw","ursell","vand",
#              "wathen","wilkinson","wing"]
#mat_names = ["chow", "erdrey", "invol", "neumann", "parallax", "pascal", "vand",
#             "smallworld", "gilbert", "chebspec"] # singular exception
#mat_names = ["binomial", "wathen", "invhilb"] # overflow?


#A = matrixdepot("blur", 2^7)
#@elapsed lu(A)
#@elapsed klu(sparse(A))
#@elapsed lu(A)
#@elapsed klu(sparse(A))
