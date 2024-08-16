#n = 2^12
#A = sprand(n, n, 0.5)
#B = Matrix(A)
#C = rand(n, n)
# @code_typed lu(A) # umfpack
# @code_typed lu(Matrix(A)) # getrf
# @code_typed lu(B) #  getrf
# @code_typed lu(C) # getrf
# @code_typed lu(Matrix(C)) #Compute getrf
# @benchmark lu($B) seconds=10 # 206.159ms
# @benchmark lu(Matrix($A)) seconds=10 # 314.668ms
# @benchmark lu($C) seconds=10 # 217.197 ms
# @benchmark lu(Matrix($C)) seconds=10 #  314.556 ms

# lu(sparse(A)) # umfpack
# lu(sparse(A)) # umfpack
# lu(sparse(B)) # umfpack
# lu(sparse(C)) # umfpack
# @benchmark lu($A) seconds=20 # umfpack, 2.123 s
# @benchmark lu(sparse($A)) seconds=20 # umfpack, 2.346 s s
# @benchmark klu($A) seconds=20 # klu, 26.511 s
# @benchmark klu(sparse($A)) seconds=20 # klu, 26.561s
# @benchmark splu($A) seconds=20 # splu, 3.840 s
# @benchmark splu(sparse($A)) seconds=20 # splu, 3.889 s

####################################################3


# A = matrixdepot("baart", 64)
# b = A * ones(64) 

# F = lu(A)
# G = lu(sparse(A))
# K = klu(sparse(A))
# S = splu(sparse(A))

# x = A \ b
# xf = F \ b
# xg = G \ b
# xk = K \ b
# xs = S \ b

# norm(A * x - b, 1)
# norm(A * xf - b, 1)
# norm(A * xg - b, 1)
# norm(A * xk - b, 1)
# norm(A * xs - b, 1)

# norm(F.L * F.U - A[F.p, :], 1)
# norm(G.L*G.U - (G.Rs .* A)[G.p, G.q], 1)
# norm(K.L * K.U + K.F - K.Rs .\ A[K.p, K.q])


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

#####################

# # Define wrappers to different LU algorithms and implementations
# function dgetrf_a(A)
#     if A isa Matrix
#         t = @elapsed L, U, p = lu(A)
#         err = norm(A[p,:] - L*U, 1)
#     else # A isa SparseMatrixCSC
#         t = @elapsed res = lu(A)
#         b = rand(size(A,1))
#         x = res \ b
#         err = norm(A * x - b, 1)
#     end
#     return t, err
# end
# function umfpack_a(A)
#     t = @elapsed res = lu(sparse(A))
#     b = rand(size(A,1))
#     x = res \ b
#     err = norm(A * x - b, 1)
#     return t, err
# end
# function klu_a(A)
#     t = @elapsed K = klu(sparse(A))
#     err = norm(K.L * K.U + K.F - K.Rs .\ A[K.p, K.q], 1)
#     return t, err
# end
# function splu_a(A)
#     t = @elapsed res = splu(sparse(A))
#     b = rand(size(A,1))
#     x = res \ b
#     err = norm(A * x - b, 1) 
#     return t, err
# end
