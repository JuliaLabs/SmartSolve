export makeplots, plot_benchmark

function makeplots(alg_path, alg_name)
    smartdb = CSV.read("$alg_path/smartdb-$alg_name.csv", DataFrame)
    fulldb = CSV.read("$alg_path/fulldb-$alg_name.csv", DataFrame)
    pattern_list = unique(smartdb[!, :pattern])
    for pattern in pattern_list
        rows = fulldb[fulldb[!, :pattern] .== pattern, :]
        ns = unique(rows[:, [:n_rows, :n_cols]])
        plot_benchmark(alg_path, alg_name, rows, ns, pattern, "log")
    end
end

function plot_benchmark(alg_path, alg_name, df, ns, mat_pattern, yaxis_type)
    # Initialize figure and variables
    f = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98),
           size = (1200, 800),
           title = "$(alg_name) Decomposition Algorithm for $(mat_pattern)",
           figure_padding = 40)
    num_experiments = unique(df[:, :n_experiment])
    num_subplots = nrow(ns)
    bar_width = 0.5
    gap = 0.6
    graphs = []
    labels = []
    label_names = []
    row_index = 1
    col_index = 1
    num_sub_horiz = ceil(Int, sqrt(num_subplots+1))
    # Add more colors if the size of the ns array is larger
    colors = [(:teal, 1.0),
              (:teal, 0.4),
              (:darkgreen, 1.0),
              (:darkgreen, 0.4),
              (:darkred, 1.0),
              (:darkred, 0.4)]
    # Iterate through each matrix size and create the graphs
    for (index, row) in enumerate(eachrow(ns))
        num_rows = row.n_rows
        num_cols = row.n_cols
        # Obtain values from dataframe for particular matrix size
        values = df[(df[!, :n_rows] .== num_rows) .&& (df[!, :n_cols] .== num_cols), :];
        algorithms = unique(values.algorithm)
        alg_time_means = []
        alg_error_means = []
        alg_time_stds = []
        alg_error_stds = []
        # Obtain means and standard deviation for each algorithm
        for alg in algorithms
            push!(alg_time_means, mean(values[values.algorithm .== alg, :time]))
            push!(alg_time_stds, min(std(values[values.algorithm .== alg, :time]), alg_time_means[end]))
            push!(alg_error_means, mean(values[values.algorithm .== alg, :error]))
            push!(alg_error_stds, min(std(values[values.algorithm .== alg, :error]), alg_error_means[end]))
        end
        num_algs = length(algorithms)
        x_vals = 1:num_algs .+ gap
        # Set up graph axes for both y axes
        ax = Axis(
            f[row_index, col_index],
            xticks = (x_vals .+ bar_width/2, collect(algorithms)),
            ylabel = "Time [s]",
            xticklabelspace = 0.2,
            ygridvisible = false,
            xgridvisible = false,
            ylabelsize = 24,
            xticklabelsize = 20,
            yticklabelsize = 18
        )
        ax2 = Axis(
            f[row_index, col_index], 
            ylabel = "Errors", 
            yaxisposition = :right,
            ygridvisible = false,
            ylabelsize = 24,
            yticklabelsize = 18,
            yscale = log10
        )
        hidespines!(ax2)
        hidexdecorations!(ax2)
        linkxaxes!(ax, ax2)
        # Set colors
        time_bar_colors = fill(colors[index*2-1], num_algs)
        error_bar_colors = fill(colors[index*2], num_algs)
        # Add graphs, labels, and errorbars
        push!(graphs, barplot!(ax, x_vals, alg_time_means, width = bar_width, color = time_bar_colors))
        push!(graphs, barplot!(ax2, x_vals .+ bar_width, alg_error_means, width = bar_width, color = error_bar_colors))
        errorbars!(ax, x_vals, alg_time_means, alg_time_stds, color = :black, linewidth = 3, whiskerwidth = 8)
        errorbars!(ax2, x_vals .+ bar_width, alg_error_means, alg_error_stds, color = :black, linewidth = 3, whiskerwidth = 8)
        push!(label_names, "$(mat_pattern): $(num_rows) x $(num_cols) Times")
        push!(label_names, "$(mat_pattern): $(num_rows) x $(num_cols) Errors")
        push!(labels, MarkerElement(color = colors[index*2-1], marker = :rect, markersize = 20,
            strokecolor = :black))
        push!(labels, MarkerElement(color = colors[index*2], marker = :rect, markersize = 20,
            strokecolor = :black))
        # Update col and row locations
        if (col_index == num_sub_horiz-1 && row_index == 1)
            row_index = 2
            col_index = 1
        elseif (col_index == num_sub_horiz)
            row_index += 1
            col_index = 1
        else
            col_index += 1
        end
    end
    colgap!(f.layout, Relative(0.05))
    rowgap!(f.layout, Relative(0.05))
    # Create legend, title, and save graphs
    leg = Legend(f[1, num_sub_horiz], labels, label_names, title = "Legend", labelsize = 20)
    leg.tellwidth = false
    Label(f[1, 1:num_sub_horiz, Top()], 
          "$(alg_name) Decomposition Algorithm for $(mat_pattern)", 
          valign = :bottom,
          font = :bold,
          padding = (0, 0, 15, 0),
          fontsize = 36)
    mat_pattern = split(mat_pattern, "/")[end]
    save("$(alg_path)/$(alg_name)_$(mat_pattern).png", f)
end