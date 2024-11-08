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

#------------------------------------------------------------------------------
function smartfeatures(df)
#Computes the contribution of each feature to reduce the error using Shapley values.
#------------------------------------------------------------------------------
#1. Calculate feature importance
    DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
    model = DecisionTreeClassifier(max_depth=3, min_samples_split=3)
    fs = [:length, :n_rows, :n_cols, :rank, :cond,
        :sparsity, :isdiag, :issymmetric, :ishermitian, :isposdef,
        :istriu, :istril]
    fs_vals = [df[:, f] for f in fs]
    X = NamedTuple{Tuple(fs)}(fs_vals)
    y = CategoricalArray(String.(df[:, :pattern]))
    mach = machine(model, X, y) |> fit!
    #------------------------------------------------------------------------------
    #PREVIOUS
    fi = feature_importances(mach)
    #------------------------------------------------------------------------------
    # #NEW
    # fit!(mach)

    # function predict_function(model, data)
    #     data_pred = DataFrame(y_pred=predict(model, data))
    #     return data_pred
    # end
    # println(df[1:50, :])
    # println(df[1:50, fs])
    # # explain = copy(df[1:300, :]) #may have to edit how many/what instances
    # # explain = select(df, Not(Symbol(outcome_name)))

    # data_shap = ShapML.shap(explain=df[1:50, fs],
    #     reference=df[:, fs],
    #     model=mach,
    #     predict_function=predict_function)

    # show(data_shap, allcols=true)

    # fi = combine(groupby(data_shap, :feature_name))[:, [:feature_name, :shap_effect]]
    #------------------------------------------------------------------------------
    fil = String.(map(x->x[1], fi))
    fiv = map(x->x[2], fi)

    nn = count(x -> x != 0, fiv) # include all nonzero features
    fi = fi[1:nn]
    fil = fil[1:nn]
    fiv = fiv[1:nn]

    println(fi)

    sparsity(x) = count(iszero, x) / length(x)
    condnumber(x::Matrix) = cond(A, 2)
    condnumber(x::SparseMatrixCSC) = cond(Array(A), 2)

#------------------------------------------------------------------------------
#2. computes the computational cost (score) of each feature.
    # Compute feature times
    ft = zeros(nn)
    c = zeros(nn)
    A = []
    for i in 1:size(df, 1)
        p = df[i, :pattern]
        n = df[i, :n_rows]
        if p in ["blur", "poisson"]
            n′ = round(Int, sqrt(n))
            n′′ = n′^2
        else
            n′ = n
            n′′ = n
        end
        println("$p, $n")
        try
            global A = convert.(Float64,matrixdepot(p, n′))
            if size(A) != (n′′, n′′)
                throw("Check matrix size: $(p), ($n, $n) vs $(size(A))")
            end
            for j in 1:nn
                ft[j] += @elapsed @eval $(Symbol(fil[j]))(A)
                c[j] += 1
            end
        catch e
            println(e)
        end
        GC.gc()
    end
    ftm = ft ./ c

    # Compute Score
    score = fiv ./ ftm

    # Plot
    plot()
    shapes = [:circle, :rect, :diamond, :star5, :utriangle]
    colors = palette(:tab10)
    for i in 1:nn
        plot!( [fiv[i]],
            [ftm[i]],
            zcolor=[score[i]],
            color=:viridis,
            seriestype = :scatter,
            thickness_scaling = 1.35,
            markersize = 7,
            markerstrokewidth = 0,
            markershapes = shapes[i],
            label=fil[i])
    end
    plot!(dpi = 300,
        label = "",
        legend=:topleft,
        yscale=:log10,
        xlabel = "Shapley-based feature importance", #changed name
        ylabel = "Time [s]",
        clims =(minimum(score), maximum(score)),
        colorbar_title = "Importance-time rate")
    savefig("featurebench.png") #change file location

#------------------------------------------------------------------------------
    #sort features by score
#     perm = sortperm(score)
#     fil[perm]

#     #feature selection algorithm
#     while i < length(fs) & (e > e_tol | t > t_tol)
#         curr_fs = fs[1:i]
#         e,t = train(mach, curr_fs, df)
#         i += 1
#     end
#     #NOT FINISHED

#     return curr_fs
end
