# SmartDiscovery

export smartsolve

function smartsolve(alg_path, alg_name, algs;
                    n_experiments = 1,
                    ns = [2^4, 2^8, 2^12],
                    mats = [])

    # Create result directory
    mkpath("$alg_path")

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
    features = smartfeatures(smartdb)
    # features = [:length, :sparsity, :condnumber]
    features_train, labels_train,
    features_test, labels_test = create_datasets(smartdb, features)
    smartmodel = train_smart_choice_model(features_train, labels_train)
    BSON.@save "$alg_path/features-$alg_name.bson" features
    BSON.@save "$alg_path/smartmodel-$alg_name.bson" smartmodel

    test_smart_choice_model(smartmodel, features_test, labels_test)

    println(typeof(test_smart_choice_model(smartmodel, features_test, labels_test)))
    println(test_smart_choice_model(smartmodel, features_test, labels_test))

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

function discover!(i, db, mat_patterns, algs, ns)
    for (j, mat_pattern) in enumerate(mat_patterns)
        for n in ns
            println("Experiment:$i, pattern number:$j, pattern:$mat_pattern, no. of cols or rows:$n.")
            flush(stdout)
            # Generate matrix
            if mat_pattern in ["blur", "poisson"]
                n′ = round(Int, sqrt(n))
                n′′ = n′^2
            else
                n′ = n
                n′′ = n
            end
            try
                A = matrixdepot(mat_pattern, n′)
                if size(A) != (n′′, n′′)
                    throw("Check matrix size: $(mat_pattern), ($n, $n) vs $(size(A))")
                end
                b = A * ones(size(A,1)) # vector required to measure error
                # Evaluate different algorithms
                for alg in algs
                    try
                        name = String(Symbol(alg))
                        t = @elapsed res = alg(A)
                        x = res \ b
                        err = norm(A * x - b, 1)
                        row  = vcat([i, mat_pattern],
                                    compute_feature_values(A),
                                    [name, t, err])
                        push!(db, row)
                    catch e
                        println("$e. $(mat_pattern), $n, $name")
                    end
                end
            catch e
                println("$e. $(mat_pattern)")
            end
            #GC.gc()
        end
    end
end

function discover!(i, db, mat_patterns, algs)
    for (j, mat_pattern) in enumerate(mat_patterns)
        println("Experiment:$i, pattern number:$j, pattern:$mat_pattern.")
        flush(stdout)
        try
            # Generate matrix
            A = matrixdepot(mat_pattern)
            b = A * ones(size(A,1)) # required to measure error
            # Evaluate different algorithms
            for alg in algs
                try
                    name = String(Symbol(alg))
                    t = @elapsed res = alg(A)
                    x = res \ b
                    err = norm(A * x - b, 1)
                    row  = vcat([i, mat_pattern],
                                compute_feature_values(A),
                                [name, t, err])
                    push!(db, row)
                catch e
                    println("$e. $(mat_pattern), $name")
                end
            end
        catch e
            println("$e. $(mat_pattern)")
        end
        GC.gc()
    end
end
