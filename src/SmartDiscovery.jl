# SmartDiscovery

export smartsolve

function smartsolve(alg_path, alg_name, algs;
                    n_experiments = 1,
                    ns = [2^4, 2^8, 2^12],
                    mats = [],
		    include_singular = false,
		    error_calc = nothing)

    # Create result directory
    run(`mkdir -p $alg_path`)

    # Save algorithms
    BSON.@save "$alg_path/algs-$alg_name.bson" algs
    
    # Define matrices
    max_entries = 10_000
    bi_patterns = mdlist(:builtin)
    sp_patterns = mdlist(sp(:) & @pred(n*m < max_entries))
    mm_patterns = mdlist(mm(:) & @pred(n*m < max_entries))

    filter!(x -> x != "parallax", bi_patterns) 
    filter!(x -> x != "invhilb", bi_patterns)
    filter!(x -> x != "neumann", bi_patterns) # Created out of memory exception
    filter!(x -> x != "clement", bi_patterns)
    filter!(x -> x != "wathen", bi_patterns)

    # Smart discovery: generate smart discovery database
    fulldb = create_empty_db()
    for i in 1:n_experiments
    	index = 1
        index = discover!(i, fulldb, bi_patterns, algs, true, error_calc, index; ns = ns)
        index = discover!(i, fulldb, sp_patterns, algs, include_singular, error_calc, index)
	    index = discover!(i, fulldb, mm_patterns, algs, include_singular, error_calc, index)
    end
    CSV.write("$alg_path/fulldb-$alg_name.csv", fulldb)

    # Smart DB: filter complete DB for faster algorithmic options
    mat_patterns = vcat(bi_patterns, sp_patterns, mm_patterns)
    push!(ns, 0)
    smartdb = get_smart_choices(fulldb, mat_patterns, ns)
    CSV.write("$alg_path/smartdb-$alg_name.csv", smartdb)

    # Smart model
    features = [:length,  :sparsity, :condnumber]
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


function generate_matrix(mat_pattern; n = 0)
    if n == 0
        return matrixdepot(mat_pattern)
    else
        if mat_pattern in ["blur", "poisson", "binomial"]
            n′ = round(Int, sqrt(n))
            n′′ = n′^2
        else
            n′ = n
            n′′ = n
        end
        A = []
        try
            A = matrixdepot(mat_pattern, n′)
            if size(A) != (n′′, n′′)
                throw("Check matrix size: $(mat_pattern), ($n, $n) vs $(size(A))")
            end
        catch e
            println("Error: $(mat_pattern), $n", typeof(e))
        end
    end
    return convert.(Float64, A)
end


function discover!(i, db, mat_patterns, algs, include_singular, error_calc, index; ns = [0])
    for (j, mat_pattern) in enumerate(mat_patterns)    
        for n in ns
	    if n == 0
            	println("Experiment:$i, Subexperiment:$index, pattern:$mat_pattern.")
	    else
		println("Experiment:$i, Subexperiment:$index, pattern:$mat_pattern, no. of cols or rows:$n.")
	    end
	    index += 1
	    flush(stdout)
            A = generate_matrix(mat_pattern; n = n)
            features = compute_feature_values(A)
            if !include_singular && features[4] < min(size(A, 1), size(A, 2))
                println("Singular Matrix!")
                flush(stdout) # Possibly fix in the future
                continue
            end
            # Generate vector b (required to measure error)
            b = A * ones(size(A, 2))
            # Evaluate different algorithms
            for alg in algs
                alg_name = String(Symbol(alg))
                if alg_name != "umfpack" && size(A, 1) != size(A, 2) # if the algorithm isn't umfpack, then don't run the algorithm if the matrix isn't square
                    continue
                end
                if count(iszero, A) / length(A) < 0.5 && alg_name == "splu" # If sparsity less than 0.5, don't run splu
                    continue
                end
                try
                    # Performs all of the castings
                    if alg_name == "dgetrf"
                        if typeof(A) <: SparseMatrixCSC
                            t_cast = @elapsed A = Matrix(A)
                        elseif typeof(A) <: SparseMatrixCSC{Bool, Int64}
                            t_cast = @elapsed A = Matrix(A)
                        elseif typeof(A) <: Symmetric
                            t_cast = @elapsed A = A.data
                        else
                            t_cast = 0
                        end
                    else
                        if typeof(A) <: Matrix
                            t_cast = @elapsed A = sparse(A)
                        elseif typeof(A) <: SparseMatrixCSC{Bool, Int64}
                            t_cast = @elapsed A = Float64.(A)
                        elseif typeof(A) <: SparseMatrixCSC{Int64, Int64}
                            t_cast = @elapsed A = Float64.(A)
                        elseif typeof(A) <: Symmetric
                            t_cast = @elapsed A = A.data
                        else
                            t_cast = 0
                        end
                    end
                    t_calc = @elapsed res = alg(A)
                                    err = default_error_calc(alg, A)
                    row = vcat([i, mat_pattern], features, [alg_name, t_cast, t_calc, err])
                    push!(db, row)
                catch e
                    if n == 0
                        println("Error: $(mat_pattern), $(size(A)), $alg_name", typeof(e))
                    else
                        println("Error: $(mat_pattern), $n, $alg_name", typeof(e))
                    end
                end
                GC.gc()
            end
        end
    end
    return index
end

function default_error_calc(alg, A)
    n = size(A, 1)
    b = rand(n, 1)
    x = alg(A) \ b
    return norm((b-A*x)/(norm(b)+norm(A)*norm(x)))
end