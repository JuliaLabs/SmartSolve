using MatrixDepot
using LinearAlgebra
using KLU
using SuperLU
using SparseArrays
using Interpolations
using DataFrames
using CSV
using PlotlyJS

# Smart discovery ##############################################################

# Define wrappers to different LU algorithms and implementations
function umfpack_a(A)
    if A isa Matrix
        t = @elapsed L, U, p = lu(A)
        err = norm(A[p,:] - L*U, 1)
    else # A isa SparseMatrixCSC
        t = @elapsed res = lu(A)
        b = rand(size(A,1))
        x = res \ b
        err = norm(A * x - b, 1)
    end
    return t, err
end
function klu_a(A)
    t = @elapsed K = klu(sparse(A))
    err = norm(K.L * K.U + K.F - K.Rs .\ A[K.p, K.q], 1)
    return t, err
end
function splu_a(A)
    t = @elapsed res = splu(sparse(A))
    b = rand(size(A,1))
    x = res \ b
    err = norm(A * x - b, 1) 
    return t, err
end
algs = [umfpack_a, klu_a, splu_a]

# Define matrices
mat_names = ["rosser", "companion", "forsythe", "grcar", "triw", "blur", "poisson",
             "heat", "kahan", "frank", "rohess", "baart", "cauchy", "circul",
             "clement", "deriv2", "dingdong", "fiedler", "foxgood", "golub","
             gravity", "hankel","hilb", "kms", "lehmer", "lotkin","magic",
             "minij", "moler","oscillate", "parter", "pei", "prolate", "randcorr",
             "rando","randsvd","sampling","shaw", "spikes", "toeplitz",
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
                    println("$e. $(mat_name), $n, $(nameof(a))")
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
                     yaxis_title="Matrix pattern, size $(n)x$(n)")
        savefig(p, "algorithms_times_$(n)_$(xaxis_type).png", width=600, height=800, scale=1.5)
    end
end

plot_benchmark(df, ns, algs, mat_names, "log")
plot_benchmark(df, ns, algs, mat_names[1:8], "linear")

for mat_name in mat_names[1:8]
    A = sparse(matrixdepot(mat_name, 2^6))
    println(A)
end 

# Generate smart choice model ##################################################

##TODO: ML model here


smart_choice = Dict()
for mat_name in mat_names
    for n in ns
        df′ = @views df[(df.mat_name .== mat_name) .&& (df.n .== n), :]
        min_time = minimum(df′.time)
        min_time_row = df′[df′.time .== min_time, :]
        push!(smart_choice, (mat_name, n) => eval(Meta.parse(min_time_row.algorithm[1])))
    end
end

################################################################################

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


#A = matrixdepot("blur", 2^5)
#b = rand(2^10)
#t = @elapsed res = lu(Matrix(A))
#x = res \ b
#err = norm(A * x - b, 1)
#t = @elapsed res = lu(A)
#x = res \ b
#err = norm(A * x - b, 1)

#A = matrixdepot("rosser", 2^10)
#b = rand(2^10)
#t = @elapsed res = klu(sparse(A))
#x = res \ b
#err = norm(A * x - b, 1)
#t = @elapsed res = klu(A)
#x = res \ b
#err = norm(A * x - b, 1)

#using Flux

#n_algs = 2 #length(algs)
#m = Chain(Conv((3,3), 1 => n_algs), GlobalMaxPool());
#n = 100
#m(rand32(n, n, 1, 1))

## Loss function and optimizer
#loss(x, y) = Flux.crossentropy(model(x), y)
#opt = ADAM()

#y_onehot = Flux.onehotbatch(algs, 0:1)

#for epoch in 1:10
#    for mat_name in mat_names
#        for n in ns
#            # Generate matrix
#            if mat_name in ["blur", "poisson"]
#                n′ = convert(Int, sqrt(n))
#            else
#                n′ = n
#            end
#            A = matrixdepot(mat_name, n′)
#            if size(A) != (n, n)
#                println("Check matrix size: $(mat_name), ($n, $n) vs $(size(A))")
#            end

#            grads = Flux.gradient(() -> loss(x, y), Flux.params(model))
#            Flux.Optimise.update!(opt, Flux.params(model), grads)
#        end
#    end
#        
#    # Calculate validation loss
#    val_loss = mean([loss(x, y) for (x, y) in val_loader])
#    println("Epoch: $epoch, Validation Loss: $val_loss")
#end

