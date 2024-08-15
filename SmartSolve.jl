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
using BSON

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

function smartsolve(alg_path, alg_name, algs)

    # Create result directory
    run(`mkdir -p $alg_path`)

    # Save algorithms
    BSON.@save "$alg_path/algs-$alg_name.bson" algs

    # Define matrices
    builtin_patterns = mdlist(:builtin)
    sp_mm_patterns = filter!(x -> x ∉ mdlist(:builtin), mdlist(:all))
    mat_patterns = builtin_patterns # [builtin_patterns; sp_mm_patterns]

    # Define matrix sizes
    #ns = [2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9, 2^10, 2^12]
    ns = [2^4, 2^8, 2^12]

    # Define number of experiments
    n_experiments = 1

    # Smart discovery: generate smart discovery database
    fulldb = create_empty_db()
    for i in 1:n_experiments
        discover!(i, fulldb, builtin_patterns, algs, ns)
        #discover!(i, fulldb, sp_mm_patterns, algs)
    end
    CSV.write("$alg_path/fulldb-$alg_name.csv", fulldb)

    # Smart DB: filter complete DB for faster algorithmic options
    smartdb = get_smart_choices(fulldb, mat_patterns, ns)
    CSV.write("$alg_path/smartdb-$alg_name.csv", smartdb)

    # Smart model
    features = [:length,  :sparsity]
    features_train, labels_train, 
    features_test, labels_test = create_datasets(smartdb, features)
    smartmodel = train_smart_choice_model(features_train, labels_train)    
    BSON.@save "$alg_path/features-$alg_name.bson" features
    BSON.@save "$alg_path/smartmodel-$alg_name.bson" smartmodel

    test_smart_choice_model(smartmodel, features_test, labels_test)
    print_tree(smartmodel, 5) # Print of the tree, to a depth of 5 nodes

    # Smart algorithm
    smartalg = """
    features_$alg_name = BSON.load("$alg_path/features-$alg_name.bson")[:features]
    smartmodel_$alg_name = BSON.load("$alg_path/smartmodel-$alg_name.bson")[:smartmodel]
    algs_$alg_name = BSON.load("$alg_path/algs-$alg_name.bson")[:algs]
    function smart$alg_name(A; features = features_$alg_name,
                        smartmodel = smartmodel_$alg_name,
                        algs = algs_$alg_name)
        fs = compute_feature_values(A; features = features)
        name = apply_tree(smartmodel, fs)
        return algs[name](A)
    end"""

    open("$alg_path/smart$alg_name.jl", "w") do file
        write(file, smartalg)
    end

    return fulldb, smartdb, smartmodel, smartalg
end

# Create a smart version of LU
alg_name = "lu"
alg_path = "smartalgs/$alg_name"
algs  = OrderedDict( "dgetrf"  => lu,
                     "umfpack" => x->lu(sparse(x)),
                     "klu"     => x->klu(sparse(x)),
                     "splu"    => x->splu(sparse(x)))
smartsolve(alg_path, alg_name, algs)

include("$alg_path/smart$alg_name.jl")

# LU vs SmartLU: time and memory usage
n = 2^14;
A = matrixdepot("poisson", round(Int, sqrt(n))); # nxn
@benchmark lu($A) seconds=200
@benchmark smartlu($A) seconds=10

# Backslash vs SmartBackslash via SmartLU: time and memory usage
b = rand(n);
@benchmark $A\$b seconds=200
@benchmark lu($A)\$b seconds=200
@benchmark smartlu($A)\$b seconds=10

# Compute errors
x = A \ b;
norm(A * x - b, 1)
x = lu(A) \ b;
norm(A * x - b, 1)
x = smartlu(A) \ b;
norm(A * x - b, 1)

# Plot results
smartdb = CSV.read("$alg_path/smartdb-$alg_name.csv", DataFrame)
fulldb = CSV.read("$alg_path/fulldb-$alg_name.csv", DataFrame)
ns = unique(smartdb[:, :n_cols])
for alg_name_i in keys(algs)
    alg_i_patterns = unique(smartdb[smartdb.algorithm .== alg_name_i, :pattern])
    plot_benchmark(alg_path, alg_name_i, fulldb, ns, algs, alg_i_patterns, "log")
end
