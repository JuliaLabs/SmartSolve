features_lu = BSON.load("smartlu//features-lu.bson")[:features]
smartmodel_lu = BSON.load("smartlu//smartmodel-lu.bson")[:smartmodel]
algs_lu = BSON.load("smartlu//algs-lu.bson")[:algs]
function smartlu(A; features = features_lu,
        smartmodel = smartmodel_lu,
        algs = algs_lu)
    fs = compute_feature_values(A; features = features)
    alg_name = apply_tree(smartmodel, fs)
    return @eval $(Symbol(alg_name))(A)
end