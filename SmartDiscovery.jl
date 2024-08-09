# Smart discovery ##############################################################

function discover!(i, db, mat_patterns, algs, ns)
    for (j, mat_pattern) in enumerate(mat_patterns)
        for n in ns
            println("Experiment:$i, pattern number:$j, pattern:$mat_pattern, no. of cols/rows:$n.")
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
                # Evaluate different algorithms
                for a in algs
                    try
                        t, err = a(A)
                        row  = vcat( [i, mat_pattern],
                                    collect(values(compute_mat_props(A))),
                                    ["$(nameof(a))", t, err])
                        push!(db, row)
                    catch e
                        println("$e. $(mat_pattern), $n, $(nameof(a))")
                    end
                end
            catch e
                println("$e. $(mat_pattern)")
            end
            GC.gc()
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
            # Evaluate different algorithms
            for a in algs
                try
                    t, err = a(A)
                    row  = vcat( [i, mat_pattern],
                                collect(values(compute_mat_props(A))),
                                ["$(nameof(a))", t, err])
                    push!(db, row)
                catch e
                    println("$e. $(mat_pattern), $(nameof(a))")
                end
            end
        catch e
            println("$e. $(mat_pattern)")
        end
        GC.gc()
    end
end
