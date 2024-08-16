# SmartDiscovery: algorithms x matrix patterns x sizes -> time x error

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
