features = OrderedDict() 
features[:length] = x -> length(x)
features[:n_rows] = x -> size(x, 1)
features[:n_cols] = x -> size(x, 2)
features[:rank] = x -> rank(x)
features[:condnumber] = x -> cond(Array(x), 2)
features[:sparsity] = x -> count(iszero, x) / length(x)
features[:isdiag] = x -> Float64(isdiag(x))
features[:issymmetric] = x -> Float64(issymmetric(x))
features[:ishermitian] = x -> Float64(ishermitian(x))
features[:isposdef] = x -> Float64(isposdef(x))
features[:istriu] = x -> Float64(istriu(x))
features[:istril] = x -> Float64(istril(x))

function compute_feature_dict(A; targetfeatures = keys(features))
    feature_dict = OrderedDict()
    for f in targetfeatures
        feature_dict[f] = features[f](A)
    end
    return feature_dict
end

function compute_feature_values(A; targetfeatures = keys(features))
    feature_vals = Float64[]
    for f in targetfeatures
        push!(feature_vals, features[f](A))
    end
    return feature_vals
end

function create_empty_db()
    df1 = DataFrame(n_experiment = Int[],
                    pattern = String[])
    features = compute_feature_dict(rand(3,3))
    column_names = keys(features)
    column_types = map(typeof, values(features))
    df2 = DataFrame(OrderedDict(k => T[] for (k, T) in zip(column_names, column_types)))
    df3 = DataFrame(algorithm = String[],
                    time = Float64[],
                    error = Float64[])
    return hcat(df1, df2, df3)
end

function get_smart_choices(db, mat_patterns, ns)
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
