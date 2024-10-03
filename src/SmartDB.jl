export compute_feature_values

all_features = OrderedDict() 
all_features[:length] = x -> length(x)
all_features[:n_rows] = x -> size(x, 1)
all_features[:n_cols] = x -> size(x, 2)
all_features[:rank] = x -> rank(x)
all_features[:condnumber] = x -> cond(Array(x), 2)
all_features[:sparsity] = x -> count(iszero, x) / length(x)
all_features[:isdiag] = x -> Float64(isdiag(x))
all_features[:issymmetric] = x -> Float64(issymmetric(x))
all_features[:ishermitian] = x -> Float64(ishermitian(x))
all_features[:isposdef] = x -> Float64(isposdef(x))
all_features[:istriu] = x -> Float64(istriu(x))
all_features[:istril] = x -> Float64(istril(x))

function compute_feature_dict(A; features = keys(all_features))
    feature_dict = OrderedDict()
    for f in features
        feature_dict[f] = all_features[f](A)
    end
    return feature_dict
end

function compute_feature_values(A; features = keys(all_features))
    feature_vals = Float64[]
    for f in features
        push!(feature_vals, all_features[f](A))
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
	    if n == 0
		db_filtered = @views db[(db.pattern .== mat_pattern), :]
            else
		db_filtered = @views db[(db.pattern .== mat_pattern) .&&
                            (db.n_cols .== n), :]
            end
	    if length(db_filtered.time) > 0
                min_time = minimum(db_filtered.time)
                min_time_row = db_filtered[db_filtered.time .== min_time, :][1, :]
                push!(db_opt, min_time_row)
            end
        end
    end
    return db_opt
end
