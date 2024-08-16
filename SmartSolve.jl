include("SmartDiscovery.jl")
include("SmartSolveDB.jl")
include("SmartChoice.jl")
include("Utils.jl")

function smartsolve(alg_path, alg_name, algs;
                    n_experiments = 1,
                    ns = [2^4, 2^8, 2^12],
                    mats = [])

    # Create result directory
    run(`mkdir -p $alg_path`)

    # Save algorithms
    BSON.@save "$alg_path/algs-$alg_name.bson" algs

    # Define matrices
    builtin_patterns = mdlist(:builtin)
    sp_mm_patterns = filter!(x -> x ∉ mdlist(:builtin), mdlist(:all))
    mat_patterns = builtin_patterns # [builtin_patterns; sp_mm_patterns]

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
        alg_name = apply_tree(smartmodel, fs)
        return @eval \$(Symbol(alg_name))(A)
    end"""

    open("$alg_path/smart$alg_name.jl", "w") do file
        write(file, smartalg)
    end

    return fulldb, smartdb, smartmodel, smartalg
end
