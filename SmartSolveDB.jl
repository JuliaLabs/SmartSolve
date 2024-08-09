function compute_mat_props(A)
    props = OrderedDict()
    props[:length] = length(A)
    props[:n_rows] = size(A, 1)
    props[:n_cols] = size(A, 2)
    props[:rank] = rank(A)
    props[:condnumber] = cond(Array(A), 2)
    props[:sparsity] = count(iszero, A) / length(A)
    props[:isdiag] = Float64(isdiag(A))
    props[:issymmetric] = Float64(issymmetric(A))
    props[:ishermitian] = Float64(ishermitian(A))
    props[:isposdef] = Float64(isposdef(A))
    props[:istriu] = Float64(istriu(A))
    props[:istril] = Float64(istril(A))
    return props
end

function create_empty_db()
    df1 = DataFrame(n_experiment = Int[],
                    pattern = String[])
    props = compute_mat_props(rand(3,3))
    column_names = keys(props)
    column_types = map(typeof, values(props))
    df2 = DataFrame(OrderedDict(k => T[] for (k, T) in zip(column_names, column_types)))
    df3 = DataFrame(algorithm = String[],
                    time = Float64[],
                    error = Float64[])
    return hcat(df1, df2, df3)
end

function compute_smart_choices(db, mat_patterns, ns)
    db_opt = create_empty_db()
    for mat_pattern in mat_patterns
        for n in ns
            db′ = @views db[(db.pattern .== mat_pattern) .&&
                            (db.n_cols .== n), :]
            if length(db′.time) > 0
                min_time = minimum(db′.time)
                min_time_row = db′[db′.time .== min_time, :][1, :]
                push!(db_opt, min_time_row)
            end
        end
    end
    return db_opt
end
