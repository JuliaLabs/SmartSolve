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

# OpenBLAS vs MKL
mkl = true
if mkl
    using MKL
end
BLAS.get_config()

include("SmartDiscovery.jl")
include("SmartSolveDB.jl")
include("SmartChoice.jl")
include("Algorithms.jl")
include("Utils.jl")

# SmartSolve workflow ########################################################

# SmartDiscovery: algorithms x matrix patterns x sizes -> time x error

# Define algorithms
algs  = [dgetrf_a, umfpack_a, klu_a, splu_a]
algs′ = [lu, x->lu(sparse(x)), klu, splu]

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
db = create_empty_db()
for i in 1:n_experiments
    discover!(i, db, builtin_patterns, algs, ns)
    #discover!(i, db, sp_mm_patterns, algs)
end
CSV.write("smartsolve.csv", db)

# Filter DB with optimal algorithms
db_opt = compute_smart_choices(db, mat_patterns, ns)

# SmartChoice model
features_train, labels_train, 
features_test, labels_test = create_datasets(db_opt)
model = train_smart_choice_model(features_train, labels_train)
test_smart_choice_model(model, features_test, labels_test)
print_tree(model, 5) # Print of the tree, to a depth of 5 nodes

# Plot benchkmark for KLU
klu_patterns = unique(db_opt[db_opt.algorithm .== "klu_a", :pattern])
plot_benchmark(db, ns, algs, klu_patterns, "log")
#plot_benchmark(db, ns, algs, mat_patterns, "log")
