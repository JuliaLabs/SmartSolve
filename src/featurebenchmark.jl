using ShapML
using RDatasets
using DataFrames
using MLJ
using CSV
using Plots
using TreeRecipe
using CategoricalArrays
using LinearAlgebra
using SparseArrays
using MatrixDepot
using ColorSchemes


# Load smart DB
df = CSV.read("smartdb-lu.csv", DataFrame)

# Compute feature importances
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
model = DecisionTreeClassifier(max_depth=3, min_samples_split=3)
fs = [:length, :n_rows, :n_cols, :rank, :cond,
      :sparsity, :isdiag, :issymmetric, :ishermitian, :isposdef,
      :istriu, :istril]
fs_vals = [df[:, f] for f in fs]
X = NamedTuple{Tuple(fs)}(fs_vals)
y = CategoricalArray(String.(df[:, :pattern]))
mach = machine(model, X, y) |> fit!
fi = feature_importances(mach)
fil = String.(map(x->x[1], fi))
fiv = map(x->x[2], fi)
nn = 4
fi = fi[1:nn]
fil = fil[1:nn]
fiv = fiv[1:nn]

sparsity(x) = count(iszero, x) / length(x)
condnumber(x::Matrix) = cond(A, 2)
condnumber(x::SparseMatrixCSC) = cond(Array(A), 2)

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

# Score
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
           #markerstrokecolor = :black,
           #markercolor = get(ColorSchemes.viridis, score[i], (minimum(score), maximum(score))),
           markershapes = shapes[i],
           label=fil[i])
end
plot!(dpi = 300,
      label = "",
      legend=:topleft,
      yscale=:log10,
      #xticks = (fi, fs),
      #ylim=(min, max),
      xlabel = "MLJ.jl-based feature importance",
      ylabel = "Time [s]",
      clims =(minimum(score), maximum(score)),
      colorbar_title = "Importance-time rate")
savefig("featurebench.png")
