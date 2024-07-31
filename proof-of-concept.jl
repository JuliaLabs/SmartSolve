using MatrixDepot
using LinearAlgebra
using KLU
using SuperLU
using SparseArrays
using Interpolations
using DataFrames
using CSV
using PlotlyJS

# OpenBLAS vs MKL
mkl = true
if mkl
    using MKL
end
BLAS.get_config()

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
mat_patterns = ["rosser", "companion", "forsythe", "grcar", "triw", "blur", "poisson",
                "heat", "kahan", "frank", "rohess", "baart", "cauchy", "circul",
                "clement", "deriv2", "dingdong", "fiedler", "foxgood", "golub",
                "gravity", "hankel","hilb", "kms", "lehmer", "lotkin","magic",
                "minij", "moler","oscillate", "parter", "pei", "prolate", "randcorr",
                "rando","randsvd","sampling","shaw", "spikes", "toeplitz",
                "tridiag","ursell", "wilkinson","wing", "hadamard", "phillips"]

# Define matrix sizes
#ns = [2^6, 2^8, 2^10, 2^12, 2^14]
ns = [2^6, 2^8, 2^10]

# Define number of experiments
n_experiments = 5

# Generate smart choice database through smart discovery
df = DataFrame(mat_pattern = String[], n = Int[], algorithm = String[], 
               time = Float64[], error = Float64[])
for i in 1:n_experiments
    println("Experiment $i")
    for mat_pattern in mat_patterns
        for n in ns
            # Generate matrix
            if mat_pattern in ["blur", "poisson"]
                n′ = convert(Int, sqrt(n))
            else
                n′ = n
            end
            A = matrixdepot(mat_pattern, n′)
            if size(A) != (n, n)
                println("Check matrix size: $(mat_pattern), ($n, $n) vs $(size(A))")
            end
            # Evaluate different algorithms
            for a in algs
                try
                    t, err = a(A)
                    push!(df, [mat_pattern, n, "$(nameof(a))", t, err])
                catch e
                    println("$e. $(mat_pattern), $n, $(nameof(a))")
                end
            end
            GC.gc()
        end
    end
end


# Show and save discovery process results ######################################

df
CSV.write("smartsolve.csv", df)

function plot_benchmark(df, ns, algs, mat_patterns, xaxis_type)
    algs_str = ["$(nameof(a))" for a in algs]
    for n in ns
        p = plot(
            [(
                ts = [
                       (
                           df′ = @views df[(df.mat_pattern .== mat_pattern) .&&
                                           (df.n .== n) .&& 
                                           (df.algorithm .== a), :];
                           if length(df′.time) > 0
                              minimum(df′.time)
                           else
                              0.0
                           end
                       )
                       for mat_pattern in reverse(mat_patterns)
                     ];
                 bar(name=a, x=ts, y=reverse(mat_patterns), orientation="h")
                ) for a in algs_str
             ])
        relayout!(p, barmode="group",
                     xaxis_type=xaxis_type,
                     xaxis_title="Time [s]",
                     yaxis_title="Matrix pattern, size $(n)x$(n)")
        savefig(p, "algorithms_times_$(n)_$(xaxis_type).png", width=600, height=800, scale=1.5)
    end
end

plot_benchmark(df, ns, algs, mat_patterns, "log")
plot_benchmark(df, ns, algs, mat_patterns[1:11], "linear")

#for mat_pattern in mat_patterns[1:11]
#    A = sparse(matrixdepot(mat_pattern, 2^6))
#    println(A)
#end 

# Generate smart choice model ##################################################

##TODO: ML model here
using Flux
using Images: imresize
using Statistics

n_patterns = length(mat_patterns)
n_cnn = 64

cnn = Flux.@autosize (n_cnn, n_cnn, 1, 1) Chain(
    Conv((5, 5), 1=>6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), _=>16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(_ => 120, relu),
    Dense(_ => 84, relu), 
    Dense(_ => n_patterns),
) |> gpu

# Loss function
loss(x, y) = Flux.logitbinarycrossentropy(x, y)

# Define optimization state object
eta = 3e-4     # learning rate
lambda = 1e-2  # for weight decay
opt_rule = OptimiserChain(WeightDecay(lambda), Adam(eta))
opt_state = Flux.setup(opt_rule, cnn)

# Create a dictionary to map patterns to indices
pattern_to_index = Dict(pattern => i for (i, pattern) in enumerate(mat_patterns))


ns = [2^6, 2^8, 2^10]

# Training 
for epoch in 1:1000
    # Create batch
    batch_size = 1024
    
    ns_batch = rand(ns, batch_size)
    
    mat_patterns_batch = rand(mat_patterns, batch_size)
    mat_pattern_indices = [pattern_to_index[mat_pattern] for mat_pattern in mat_patterns_batch]
    onehot_encoded = Flux.onehotbatch(mat_pattern_indices, 1:length(mat_patterns))
    
    As = []
    for j in 1:batch_size
        # Generate matrix
        mat_pattern = mat_patterns_batch[j]
        n = ns_batch[j]
        if mat_pattern in ["blur", "poisson"]
            n′ = convert(Int, sqrt(n))
        else
            n′ = n
        end
        A = matrixdepot(mat_pattern, n′)
        if size(A) != (n, n)
            println("Check matrix size: $(mat_pattern), ($n, $n) vs $(size(A))")
        end

        row_start = rand(1:n-n_cnn+1)
        col_start = rand(1:n-n_cnn+1)
        A = A[row_start:row_start+n_cnn-1, col_start:col_start+n_cnn-1]

        #push!(As, imresize(A, (n_cnn, n_cnn), method=Linear()))
        push!(As, A)
    end
    A′ = cat(As..., dims=4)

    grads = Flux.gradient(cnn -> loss(cnn(A′), onehot_encoded), cnn)
    Flux.update!(opt_state, cnn, grads[1])

    # Checks
    ŷ = cnn(A′)
    y = onehot_encoded
    train_batch_loss = loss(ŷ, y)
    train_bacth_acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    println("Epoch: $epoch, training batch loss: $(train_batch_loss), training batch acc: $train_bacth_acc")

end

# Predict
function predict(A, cnn)
    A′ = imresize(A, (n_cnn, n_cnn))
    A′ = reshape(A′, (n_cnn, n_cnn, 1, 1), method=Linear())
    return cnn(A′)
end

#rand_pattern = rand(mat_patterns)
#rand_n = ns_batch = rand(ns)
#A = matrixdepot(rand_pattern, rand_n)
#pattern_to_index[rand_pattern]
#predict(A, cnn)
#plot(reshape(predict(A, cnn), (46)))


smart_choice = Dict()
for mat_pattern in mat_patterns
    for n in ns
        df′ = @views df[(df.mat_name .== mat_pattern) .&& (df.n .== n), :]
        min_time = minimum(df′.time)
        min_time_row = df′[df′.time .== min_time, :]
        push!(smart_choice, (mat_pattern, n) => eval(Meta.parse(min_time_row.algorithm[1])))
    end
end

################################################################################

# All matrices
# mat_patterns =  ["baart","binomial","blur","cauchy","chebspec","chow","circul",
#              "clement","companion","deriv2","dingdong","erdrey","fiedler",
#              "forsythe","foxgood","frank","gilbert","golub","gravity","grcar",
#              "hadamard","hankel","heat","hilb","invhilb","invol","kahan","kms",
#              "lehmer","lotkin","magic","minij","moler","neumann","oscillate",
#              "parallax","parter","pascal","pei","phillips","poisson","prolate",
#              "randcorr","rando","randsvd","rohess","rosser","sampling","shaw",
#              "smallworld","spikes","toeplitz","tridiag","triw","ursell","vand",
#              "wathen","wilkinson","wing"]
#mat_patterns = ["chow", "erdrey", "invol", "neumann", "parallax", "pascal", "vand",
#             "smallworld", "gilbert", "chebspec"] # singular exception
#mat_patterns = ["binomial", "wathen", "invhilb"] # overflow?


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


###############################################################################

#using Flux
#using Images: imresize
#using Statistics

#n_patterns = length(mat_patterns)
#n_cnn = 28

#cnn = Flux.@autosize (n_cnn, n_cnn, 1, 1) Chain(
#    Conv((5, 5), 1=>6, sigmoid),
#    MaxPool((2, 2)),
#    Conv((5, 5), _=>16, sigmoid),
#    MaxPool((2, 2)),
#    Flux.flatten,
#    Dense(_ => 120, sigmoid),
#    Dense(_ => 84, sigmoid), 
#    Dense(_ => n_patterns),
#) |> gpu

## Loss function
#loss(x, y) = Flux.logitbinarycrossentropy(x, y)
##loss(x, y) = Flux.mse(x, y)

## Define optimization state object
#eta = 3e-4     # learning rate
#lambda = 1e-2  # for weight decay
#opt_rule = OptimiserChain(WeightDecay(lambda), Adam(eta))
#opt_state = Flux.setup(opt_rule, cnn)

## Create a dictionary to map patterns to indices
#pattern_to_index = Dict(pattern => i for (i, pattern) in enumerate(mat_patterns))

## Training 
#for epoch in 1:100
#    # Create batch
#    batch_size = 32
#    
#    ns_batch = rand(ns, batch_size)
#    
#    mat_patterns_batch = rand(mat_patterns, batch_size)
#    mat_pattern_indices = [pattern_to_index[mat_pattern] for mat_pattern in mat_patterns_batch]
#    onehot_encoded = Flux.onehotbatch(mat_pattern_indices, 1:length(mat_patterns))
#    
#    As = []
#    for j in 1:batch_size
#        # Generate matrix
#        mat_pattern = mat_patterns_batch[j]
#        n = ns_batch[j]
#        if mat_pattern in ["blur", "poisson"]
#            n′ = convert(Int, sqrt(n))
#        else
#            n′ = n
#        end
#        A = matrixdepot(mat_pattern, n′)
#        if size(A) != (n, n)
#            println("Check matrix size: $(mat_pattern), ($n, $n) vs $(size(A))")
#        end
#        push!(As, imresize(A, (n_cnn, n_cnn), method=Linear()))
#    end
#    A′ = cat(As..., dims=4)

#    grads = Flux.gradient(cnn -> loss(cnn(A′), onehot_encoded), cnn)
#    Flux.update!(opt_state, cnn, grads[1])

#    # Checks
#    ŷ = cnn(A′)
#    y = onehot_encoded
#    train_batch_loss = loss(ŷ, y)
#    train_bacth_acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
#    println("Epoch: $epoch, training batch loss: $(train_batch_loss), training batch acc: $train_bacth_acc")

#end

## Predict
#function predict(A, cnn)
#    A′ = imresize(A, (n_cnn, n_cnn))
#    A′ = reshape(A′, (n_cnn, n_cnn, 1, 1), method=Linear())
#    return cnn(A′)
#end

#rand_pattern = rand(mat_patterns)
#rand_n = ns_batch = rand(ns)
#A = matrixdepot(rand_pattern, rand_n)
#pattern_to_index[rand_pattern]
#predict(A, cnn)
#plot(reshape(predict(A, cnn), (46)))


#smart_choice = Dict()
#for mat_pattern in mat_patterns
#    for n in ns
#        df′ = @views df[(df.mat_name .== mat_pattern) .&& (df.n .== n), :]
#        min_time = minimum(df′.time)
#        min_time_row = df′[df′.time .== min_time, :]
#        push!(smart_choice, (mat_pattern, n) => eval(Meta.parse(min_time_row.algorithm[1])))
#    end
#end

