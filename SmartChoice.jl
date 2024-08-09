# Smart choice model: decision tree based selection

function create_datasets(db, features)
    # Split dataset into training and test
    n_db = size(db, 1)
    n_train = round(Int, n_db * 0.8)
    n_test = n_db - n_train
    inds = randperm(n_db)
    labels = [:algorithm]
    features_train = @views Matrix(db[inds[1:n_train], features])
    labels_train = @views vec(Matrix(db[inds[1:n_train], labels]))
    features_test = @views Matrix(db[inds[1:n_test], features])
    labels_test = @views vec(Matrix(db[inds[1:n_test], labels]))
    return  features_train, labels_train,
            features_test, labels_test
end

function train_smart_choice_model(features_train, labels_train)
    # Train full-tree classifier
    model = build_tree(labels_train, features_train)
    # Prune tree: merge leaves having >= 90% combined purity (default: 100%)
    model = prune_tree(model, 0.9)
    return model
end

function test_smart_choice_model(model, features_test, labels_test)
    # Apply learned model
    apply_tree(model, features_test[1, :])

    # Generate confusion matrix, along with accuracy and kappa scores
    preds_test = apply_tree(model, features_test)
    
    return DecisionTree.confusion_matrix(labels_test, preds_test)
end


############################################################################
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
#            a = :dgetrf_a
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

# mat_patterns = ["rosser", "companion", "forsythe", "grcar", "triw", "blur", "poisson",
#                 "heat", "kahan", "frank", "rohess", "baart", "cauchy", "circul",
#                 "clement", "deriv2", "dingdong", "fiedler", "foxgood", "golub",
#                 "gravity", "hankel","hilb", "kms", "lehmer", "lotkin","magic",
#                 "minij", "moler","oscillate", "parter", "pei", "prolate", "randcorr",
#                 "rando","randsvd","sampling","shaw", "spikes", "toeplitz",
#                 "tridiag","ursell", "wilkinson","wing", "hadamard", "phillips"]
#