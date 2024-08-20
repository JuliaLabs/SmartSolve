# Smart choice model: decision tree based selection

function create_datasets(db, features)
    # Split dataset into training and test
    n_db = size(db, 1)
    n_train = round(Int, n_db * 0.8)
    n_test = n_db - n_train
    inds = randperm(n_db)
    labels = [:algorithm]
    features_train = @views Matrix(db[inds[1:n_train], features])
    labels_train = @views vec(Matrix(db[inds[1:n_train], labels]))
    features_test = @views Matrix(db[inds[1:n_test], features])
    labels_test = @views vec(Matrix(db[inds[1:n_test], labels]))
    return  features_train, labels_train,
            features_test, labels_test
end

function train_smart_choice_model(features_train, labels_train)
    # Train full-tree classifier
    n_subfeat = 0
    n_feat = size(features_train[1, :], 1)
    model = build_tree(labels_train, features_train, n_subfeat, n_feat)
    # Prune tree: merge leaves having >= 90% combined purity (default: 100%)
    # model = prune_tree(model, 0.9)
    return model
end

function test_smart_choice_model(model, features_test, labels_test)
    # Apply learned model
    apply_tree(model, features_test[1, :])

    # Generate confusion matrix, along with accuracy and kappa scores
    preds_test = apply_tree(model, features_test)
    
    return DecisionTree.confusion_matrix(labels_test, preds_test)
end

