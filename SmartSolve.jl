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

function smartsolve(name, algs)

    # Define matrices
    builtin_patterns = mdlist(:builtin)
    sp_mm_patterns = filter!(x -> x ∉ mdlist(:builtin), mdlist(:all))
    mat_patterns = builtin_patterns # [builtin_patterns; sp_mm_patterns]

    # Define matrix sizes
    #ns = [2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9, 2^10]
    ns = [2^6, 2^8, 2^10]

    # Define number of experiments
    n_experiments = 1

    # Generate smart discovery database
    fulldb = create_empty_db()
    for i in 1:n_experiments
        discover!(i, fulldb, builtin_patterns, algs, ns)
        #discover!(i, fulldb, sp_mm_patterns, algs)
    end

    # Filter full DB with optimal choices in terms of performance
    smartdb = get_smart_choices(fulldb, mat_patterns, ns)

    # SmartChoice model
    features = [:length,  :sparsity]
    features_train, labels_train, 
    features_test, labels_test = create_datasets(smartdb, features)
    smartmodel = train_smart_choice_model(features_train, labels_train)
    test_smart_choice_model(smartmodel, features_test, labels_test)
    print_tree(smartmodel, 5) # Print of the tree, to a depth of 5 nodes

    function smartalg(A, smartmodel, algs)
        features = compute_feature_values(A; targetfeatures = [:length,  :sparsity])
        name = apply_tree(smartmodel, features)
        return algs[name](A)
    end

    return fulldb, smartdb, smartmodel, smartalg
end

# Create a smart version of LU
name = "LU"
algs  = OrderedDict( "dgetrf"  => lu,
                     "umfpack" => x->lu(sparse(x)),
                     "klu"     => x->klu(sparse(x)),
                     "splu"    => x->splu(sparse(x)))
fulldb, smartdb, smartmodel, smartlu = smartsolve(name, algs)

# Benchmark speed
A = matrixdepot("rosser", 2^10)
@benchmark res_lu = lu($A)
@benchmark res_smartlu = smartlu($A, $smartmodel, $algs)

# Compute errors
b = rand(2^10)
x = lu(A) \ b
norm(A * x - b, 1)
x = smartlu(A, smartmodel, algs) \ b
norm(A * x - b, 1)

# Save and plot
CSV.write("fulldb-$name.csv", fulldb)
CSV.write("smartdb-$name.csv", smartdb)

klu_patterns = unique(smartdb[smartdb.algorithm .== "klu_a", :pattern])
plot_benchmark(fulldb, ns, algs, klu_patterns, "log")
