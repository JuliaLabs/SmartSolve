# Smart choice model: decision tree based selection

function create_datasets(db, features)
    # Split dataset into training and test
    n_db = size(db, 1)
    n_train = round(Int, n_db * 0.5)
    n_test = n_db - n_train
    inds = randperm(n_db)
    labels = [:algorithm]
    # features_train = @views Matrix(db[inds[1:n_train], features])
    # labels_train = @views vec(Matrix(db[inds[1:n_train], labels]))
    # features_test = @views Matrix(db[inds[1:n_test], features])
    # labels_test = @views vec(Matrix(db[inds[1:n_test], labels]))
    features_train = @views db[inds[1:n_train], features]
    labels_train = @views vec(Matrix(db[inds[1:n_train], labels]))
    features_test = @views db[inds[1:n_test], features]
    labels_test = @views vec(Matrix(db[inds[1:n_test], labels]))
    return  features_train, labels_train,
            features_test, labels_test
end

function train_smart_choice_model(features_train, labels_train, features)
    # Train full-tree classifier
    # n_subfeat = 0
    # n_feat = size(features_train[1, :], 1)
    # model = build_tree(labels_train, features_train, n_subfeat, n_feat)
    # # Prune tree: merge leaves having >= 90% combined purity (default: 100%)
    # model = prune_tree(model, 0.9)
    DecisionTreeClassifier = @load DecisionTreeClassifier pkg = DecisionTree verbosity=0
    model = DecisionTreeClassifier(max_depth=3, min_samples_split=3)
    fs_vals = [features_train[:, f] for f in features]
    X = NamedTuple{Tuple(features)}(fs_vals)
    y = CategoricalArray(String.(labels_train))
    mach = machine(model, X, y) |> MLJ.fit!
    return mach
end

function test_smart_choice_model(mach, features_test, labels_test, features)
    # # Apply learned model
    # apply_tree(model, features_test[1, :])

    # # Generate confusion matrix, along with accuracy and kappa scores
    # preds_test = apply_tree(model, features_test)

    # # return DecisionTree.confusion_matrix(labels_test, preds_test)
    # return log_loss(labels_test, preds_test)
    # features = [:length, :n_rows, :n_cols, :rank, :condnumber,
    #     :sparsity, :isdiag, :issymmetric, :ishermitian, :isposdef,
    #     :istriu, :istril]
    fs_vals = [features_test[:, f] for f in features]
    X = NamedTuple{Tuple(features)}(fs_vals)
    y = CategoricalArray(String.(labels_test))
    yhat = predict_mode(mach, X)
    return ConfusionMatrix()(yhat, y)
end
