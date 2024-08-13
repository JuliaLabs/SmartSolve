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
using DecisionTree
using Random
using JLD2
using BenchmarkTools

# OpenBLAS vs MKL
mkl = true
if mkl
    using MKL
end
BLAS.get_config()

include("SmartDiscovery.jl")
include("SmartSolveDB.jl")
include("SmartChoice.jl")
include("Utils.jl")

# SmartSolve workflow ########################################################

function smartsolve(path, name, algs)

    # Create result directory
    run(`mkdir -p $path/$name`)

    # Save algorithms
    jldsave("$path/$name/algs-$name.jld2"; algs)

    # Define matrices
    builtin_patterns = mdlist(:builtin)
    sp_mm_patterns = filter!(x -> x ∉ mdlist(:builtin), mdlist(:all))
    mat_patterns = builtin_patterns # [builtin_patterns; sp_mm_patterns]

    # Define matrix sizes
    #ns = [2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9, 2^10, 2^12]
    ns = [2^4, 2^8]

    # Define number of experiments
    n_experiments = 1

    # Smart discovery: generate smart discovery database
    fulldb = create_empty_db()
    for i in 1:n_experiments
        discover!(i, fulldb, builtin_patterns, algs, ns)
        #discover!(i, fulldb, sp_mm_patterns, algs)
    end
    CSV.write("$path/$name/fulldb-$name.csv", fulldb)

    # Smart DB: filter complete DB for faster algorithmic options
    smartdb = get_smart_choices(fulldb, mat_patterns, ns)
    CSV.write("$path/$name/smartdb-$name.csv", smartdb)

    # Smart model
    features = [:length,  :sparsity]
    features_train, labels_train, 
    features_test, labels_test = create_datasets(smartdb, features)
    smartmodel = train_smart_choice_model(features_train, labels_train)    
    jldsave("$path/$name/features-$name.jld2"; features)
    jldsave("$path/$name/smartmodel-$name.jld2"; smartmodel)

    test_smart_choice_model(smartmodel, features_test, labels_test)
    print_tree(smartmodel, 5) # Print of the tree, to a depth of 5 nodes

    # Smart algorithm
    smartalg = """
    features_$name = load("$path/$name/features-$name.jld2")["features"]
    smartmodel_$name = load("$path/$name/smartmodel-$name.jld2")["smartmodel"]
    algs_$name = load("$path/$name/algs-$name.jld2")["algs"]
    function smart$name(A; features = features_$name,
                        smartmodel = smartmodel_$name,
                        algs = algs_$name)
        fs = compute_feature_values(A; features = features)
        name = apply_tree(smartmodel, fs)
        return algs[name](A)
    end"""

    open("$path/$name/smart$name.jl", "w") do file
        write(file, smartalg)
    end

    return fulldb, smartdb, smartmodel, smartalg
end

# Create a smart version of LU
path = "smartalgs"
name = "lu"
algs  = OrderedDict( "dgetrf"  => lu,
                     "umfpack" => x->lu(sparse(x)),
                     "klu"     => x->klu(sparse(x)),
                     "splu"    => x->splu(sparse(x)))
smartsolve(path, name, algs)

include("$path/$name/smart$name.jl")

# Benchmark speed
n = 2^10
A = matrixdepot("blur", round(Int, sqrt(n))) # nxn
@benchmark lu($A)
@benchmark smartlu($A)

# Compute errors
b = rand(n)
x = lu(A) \ b
norm(A * x - b, 1)
x = smartlu(A) \ b
norm(A * x - b, 1)

# Plot KLU results
#klu_patterns = unique(smartdb[smartdb.algorithm .== "klu_a", :pattern])
#plot_benchmark(fulldb, ns, algs, klu_patterns, "log")
