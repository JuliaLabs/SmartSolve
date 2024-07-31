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

# OpenBLAS vs MKL
mkl = true
if mkl
    using MKL
end
BLAS.get_config()

# Auxiliary functions ##########################################################

function compute_mat_props(A)
    props = OrderedDict()
    props[:length] = length(A)
    props[:n_rows] = size(A, 1)
    props[:n_cols] = size(A, 2)
    props[:rank] = rank(A)
    props[:condnumber] = cond(Array(A), 2)
    props[:sparsity] = count(iszero, A) / length(A)
    props[:isdiag] = Float64(isdiag(A))
    props[:issymmetric] = Float64(issymmetric(A))
    props[:ishermitian] = Float64(ishermitian(A))
    props[:isposdef] = Float64(isposdef(A))
    props[:istriu] = Float64(istriu(A))
    props[:istril] = Float64(istril(A))
    return props
end

function create_dataframe()
    df1 = DataFrame(n_experiment = Int[],
                    pattern = String[])
    props = compute_mat_props(rand(3,3))
    column_names = keys(props)
    column_types = map(typeof, values(props))
    df2 = DataFrame(OrderedDict(k => T[] for (k, T) in zip(column_names, column_types)))
    df3 = DataFrame(algorithm = String[],
                    time = Float64[],
                    error = Float64[])
    return hcat(df1, df2, df3)
end

function plot_benchmark(df, ns, algs, mat_patterns, xaxis_type)
    algs_str = ["$(nameof(a))" for a in algs]
    for n in ns
        p = plot(
            [(
                ts = [
                       (
                           df′ = @views df[(df.pattern .== mat_pattern) .&&
                                           (df.n_cols .== n) .&& 
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
ns = [2^5, 2^6, 2^7, 2^8, 2^9, 2^10]

# Define number of experiments
n_experiments = 1

# Generate smart choice database through smart discovery #######################
df = create_dataframe()
for i in 1:n_experiments
    println("Experiment $i")
    for mat_pattern in mat_patterns
        for n in ns
            # Generate matrix
            if mat_pattern in ["blur", "poisson"]
                n′ = round(Int, sqrt(n))
                n′′ = n′^2
            else
                n′ = n
                n′′ = n
            end
            A = matrixdepot(mat_pattern, n′)
            if size(A) != (n′′, n′′)
                println("Check matrix size: $(mat_pattern), ($n, $n) vs $(size(A))")
            end
            # Evaluate different algorithms
            for a in algs
                try
                    t, err = a(A)
                    row  = vcat( [i, mat_pattern],
                                 collect(values(compute_mat_props(A))),
                                 ["$(nameof(a))", t, err])
                    push!(df, row)
                catch e
                    println("$e. $(mat_pattern), $n, $(nameof(a))")
                end
            end
            GC.gc()
        end
    end
end


# Show and save discovery dataframe ############################################

df
CSV.write("smartsolve.csv", df)

plot_benchmark(df, ns, algs, mat_patterns, "log")
plot_benchmark(df, ns, algs, mat_patterns[1:11], "linear")


# Generate smart choice model ##################################################

# Select rows with min time per matrix pattern and size ########################
df_opt = create_dataframe()
for mat_pattern in mat_patterns
    for n in ns
        df′ = @views df[(df.pattern .== mat_pattern) .&& (df.n_cols .== n), :]
        if length(df′.time) > 0
            min_time = minimum(df′.time)
            min_time_row = df′[df′.time .== min_time, :][1, :]
            push!(df_opt, min_time_row)
        end
    end
end

# Decision tree based selection
using DecisionTree
using ScikitLearn.CrossValidation: cross_val_score
using Random

# Split dataset into training and test
n_df = size(df_opt, 1)
n_train = round(Int, n_df * 0.8)
n_test = n_df - n_train
inds = randperm(n_df)
features = [:length, :rank, :condnumber, :sparsity, :isdiag, :issymmetric,
            :ishermitian, :isposdef, :istriu, :istril]
labels = [:algorithm]
features_train = @views Matrix(df_opt[inds[1:n_train], features])
labels_train = @views vec(Matrix(df_opt[inds[1:n_train], labels]))
features_test = @views Matrix(df_opt[inds[1:n_test], features])
labels_test = @views vec(Matrix(df_opt[inds[1:n_test], labels]))

# Define and fit decision tree classifier
model = DecisionTreeClassifier(max_depth=4)
fit!(model, features_train, labels_train)

# Test
s = 0
klu_ok = 0
klu_fail = 0
umfpack_ok = 0
umfpack_fail = 0
n_test = size(features_test, 1)
for i in 1:n_test
    pred_label = predict(model, features_test[i, :])
    if labels_test[i] == pred_label
        s += 1
    end
    if labels_test[i] == "klu_a"
        if labels_test[i] == pred_label
            klu_ok += 1
        else
            klu_fail += 1
        end
    end
    if labels_test[i] == "umfpack_a"
        if labels_test[i] == pred_label
            umfpack_ok += 1
        else
            umfpack_fail += 1
        end
    end
end
s / n_test
klu_ok / sum(labels_test .== "klu_a")
klu_fail / sum(labels_test .== "klu_a")
umfpack_ok / sum(labels_test .== "umfpack_a")
umfpack_fail / sum(labels_test .== "umfpack_a")


##########3

#n_features = 9
#n_samples = 600
#features = Matrix{Float64}(undef, n_samples, n_features)
#labels = []
#for i in 1:n_samples
#    println("Sample: $i")
#    # Generate matrix
#    mat_pattern = rand(mat_patterns)
#    n = rand(ns)
#    if mat_pattern in ["blur", "poisson"]
#        n′ = convert(Int, sqrt(n))
#    else
#        n′ = n
#    end
#    A = matrixdepot(mat_pattern, n′)
#    if size(A) != (n, n)
#        println("Check matrix size: $(mat_pattern), ($n, $n) vs $(size(A))")
#    end
#    # Save label
#    push!(labels, "$(smart_choice[(mat_pattern, n)])")
#    # Compute and save features
#    size_f = n^2
#    rank_f = rank(A)
#    sparsity_f = count(iszero, A) / length(A)
#    condnumber_f = cond(Array(A), 2)
#    issymmetric_f = Float64(issymmetric(A))
#    istril_f = Float64(istril(A))
#    istriu_f = Float64(istriu(A))
#    isdiag_f = Float64(isdiag(A))
#    isposdef_f = Float64(isposdef(A))
#    features[i, :] = [size_f, rank_f, sparsity_f, condnumber_f, issymmetric_f,
#                      istril_f, istriu_f, isdiag_f, isposdef_f]
#end


#model = DecisionTreeClassifier(max_depth=4)
#fit!(model, features, labels)

## Validate using ScikitLearn

#accuracy = cross_val_score(model, features, labels, cv=3)

## Naive validation
#s = 0
#klu_ok = 0
#klu_fail = 0
#umfpack_ok = 0
#umfpack_fail = 0
#for i in 1:n_samples
#    pred_label = predict(model, features[i, :])
#    if labels[i] == pred_label
#        s += 1
#    end
#    if labels[i] == "klu_a"
#        if labels[i] == pred_label
#            klu_ok += 1
#        else
#            klu_fail += 1
#        end
#    end
#    if labels[i] == "umfpack_a"
#        if labels[i] == pred_label
#            umfpack_ok += 1
#        else
#            umfpack_fail += 1
#        end
#    end
#end
#s / n_samples
#klu_ok / sum(labels .== "klu_a")
#klu_fail / sum(labels .== "klu_a")
#umfpack_ok / sum(labels .== "umfpack_a")
#umfpack_fail / sum(labels .== "umfpack_a")

##############

#function closest_power_of_two(N::Int)
#    # Get the next and previous powers of two
#    next_pow2 = nextpow(2, N)
#    prev_pow2 = prevpow(2, N)
#    
#    # Return the closest one
#    if (next_pow2 - N) < (N - prev_pow2)
#        return next_pow2
#    else
#        return prev_pow2
#    end
#end

#function closest_even_power_of_two(N::Int)
#    # Get the next and previous powers of two
#    next_pow2 = nextpow(2, N)
#    prev_pow2 = prevpow(2, N)
#    
#    # Ensure the closest power of two is even
#    if next_pow2 % 2 != 0
#        next_pow2 *= 2
#    end
#    if prev_pow2 % 2 != 0
#        prev_pow2 *= 2
#    end
#    
#    # Return the closest one
#    if (next_pow2 - N) < (N - prev_pow2)
#        return next_pow2
#    else
#        return prev_pow2
#    end
#end

#ns = [2^5, 2^7, 2^9]
#s = 0
#klu_ok = 0
#klu_fail = 0
#umfpack_ok = 0
#umfpack_fail = 0
#features = Matrix{Float64}(undef, n_samples, n_features)
#labels = []
#for i in 1:n_samples

#    # Generate matrix
#    mat_pattern = rand(mat_patterns)
#    n = rand(ns)
#    println("Sample: $i, pattern: $mat_pattern, n:$n")
#    if mat_pattern in ["blur", "poisson"]
#        n′ = round(Int, sqrt(n))
#        n  = n′^2
#    else
#        n′ = n
#    end
#    A = matrixdepot(mat_pattern, n′)
#    if size(A) != (n, n)
#        println("Check matrix size: $(mat_pattern), ($n, $n) vs $(size(A))")
#    end
#    # Save label
#    m = closest_even_power_of_two(n)
#    push!(labels, "$(smart_choice[(mat_pattern, m)])")
#    
#    # Compute and save features
#    size_f = n^2
#    rank_f = rank(A)
#    sparsity_f = count(iszero, A) / length(A)
#    condnumber_f = cond(Array(A), 2)
#    issymmetric_f = Float64(issymmetric(A))
#    istril_f = Float64(istril(A))
#    istriu_f = Float64(istriu(A))
#    isdiag_f = Float64(isdiag(A))
#    isposdef_f = Float64(isposdef(A))
#    features[i, :] = [size_f, rank_f, sparsity_f, condnumber_f, issymmetric_f,
#                      istril_f, istriu_f, isdiag_f, isposdef_f]
#    # Predict
#    pred_label = predict(model, features[i, :])
#    
#    # Compute errors
#    if labels[i] == pred_label
#        s += 1
#    end
#    if labels[i] == "klu_a"
#        if labels[i] == pred_label
#            klu_ok += 1
#        else
#            klu_fail += 1
#        end
#    end
#    if labels[i] == "umfpack_a"
#        if labels[i] == pred_label
#            umfpack_ok += 1
#        else
#            umfpack_fail += 1
#        end
#    end 

#end
#s / n_samples
#klu_ok / sum(labels .== "klu_a")
#klu_fail / sum(labels .== "klu_a")
#umfpack_ok / sum(labels .== "umfpack_a")
#umfpack_fail / sum(labels .== "umfpack_a")

# pretty print of the tree, to a depth of 5 nodes (optional)
#print_tree(model, 5)
# apply learned model
#predict(model, [5.9,3.0,5.1,1.9])
# get the probability of each label
#predict_proba(model, [5.9,3.0,5.1,1.9])
#println(get_classes(model)) # returns the ordering of the columns in predict_proba's output



### CNN based selection
#using Flux
#using Images: imresize
#using Statistics
#using CUDA
#using cuDNN

#n_patterns = length(mat_patterns)
#n_cnn = 64

#cnn = Flux.@autosize (n_cnn, n_cnn, 1, 1) Chain(
#    Conv((5, 5), 1=>6, relu),
#    MaxPool((2, 2)),
#    Conv((5, 5), _=>16, relu),
#    MaxPool((2, 2)),
#    Flux.flatten,
#    Dense(_ => 120, relu),
#    Dense(_ => 84, relu), 
#    Dense(_ => n_patterns),
#) 
#cnn = cnn |> gpu

## Loss function
#loss(x, y) = Flux.logitbinarycrossentropy(x, y)

## Define optimization state object
#eta = 3e-4     # learning rate
#lambda = 1e-2  # for weight decay
#opt_rule = OptimiserChain(WeightDecay(lambda), Adam(eta))
#opt_state = Flux.setup(opt_rule, cnn)

## Create a dictionary to map patterns to indices
#pattern_to_index = Dict(pattern => i for (i, pattern) in enumerate(mat_patterns))


#ns = [2^6, 2^8, 2^10]

## Training 
#for epoch in 1:1000
#    # Create batch
#    batch_size = 1024
#    
#    ns_batch = rand(ns, batch_size)
#    
#    mat_patterns_batch = rand(mat_patterns, batch_size)
#    mat_pattern_indices = [pattern_to_index[mat_pattern] for mat_pattern in mat_patterns_batch]
#    onehot_encoded = Flux.onehotbatch(mat_pattern_indices, 1:length(mat_patterns)) |> gpu
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

#        row_start = rand(1:n-n_cnn+1)
#        col_start = rand(1:n-n_cnn+1)
#        A = A[row_start:row_start+n_cnn-1, col_start:col_start+n_cnn-1]

#        #push!(As, imresize(A, (n_cnn, n_cnn), method=Linear()))
#        push!(As, A)
#    end
#    A′ = cat(As..., dims=4) |> gpu
#    
#    GC.gc()

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

################################################################################


# This model associates matrix size and pattern with best algorithm
#smart_choice = Dict()
#for mat_pattern in mat_patterns
#    for n in ns
#        df′ = @views df[(df.pattern .== mat_pattern) .&& (df.n_cols .== n), :]
#        if length(df′.time) > 0
#            min_time = minimum(df′.time)
#            min_time_row = df′[df′.time .== min_time, :]
#            a = eval(Meta.parse(min_time_row.algorithm[1]))
#        else
#            a = :umfpack_a
#        end
#        push!(smart_choice, (mat_pattern, n) => a)
#    end
#end
#smart_choice[("poisson", 1024)]


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

