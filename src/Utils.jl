export makeplots, plot_benchmark

function makeplots(alg_path, alg_name; tick_label_size=30, label_size=36, title_size=48)
    smartdb = CSV.read("$alg_path/smartdb-$alg_name.csv", DataFrame)
    fulldb = CSV.read("$alg_path/fulldb-$alg_name.csv", DataFrame)
    pattern_list = unique(smartdb[!, :pattern])
    for pattern in pattern_list
        df = fulldb[fulldb[!, :pattern] .== pattern, :]
        ns = unique(df[:, [:n_rows, :n_cols]])
        for n in eachrow(ns)
            num_rows = n.n_rows
            num_cols = n.n_cols
            values = df[(df[!, :n_rows] .== num_rows) .&& (df[!, :n_cols] .== num_cols), :];
            plot_benchmark(alg_path, alg_name, values, num_rows, num_cols, pattern)
        end
    end
end

function plot_benchmark(alg_path, alg_name, df, num_rows, num_cols, mat_pattern; tick_label_size=30, label_size=36, title_size=48)
    # Initialize figure and variables
    f = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98),
           size = (2400, 800),
           title = "$(uppercase(alg_name)) Decomposition Algorithm for $(mat_pattern): $(num_rows) x $(num_cols)",
           figure_padding = (40, 80, 10, 10))
    num_experiments = unique(df[:, :n_experiment])
    # Add more colors if the size of the ns array is larger
    colors = [(:teal, 1.0),
              (:darkgreen, 1.0),
              (:darkred, 1.0),
              (:purple, 1.0)]
    shapes = [
        :circle,
        :diamond,
        :cross,
        :star5
    ]

    # Get all the algorithms
    algorithms = unique(df.algorithm)
    num_algs = length(algorithms)
    cast_time_means = []
    cast_time_stds = []
    alg_time_means = []
    alg_time_means = []
    alg_error_means = []
    alg_time_stds = []
    alg_error_stds = []

    # Obtain means and standard deviation for each algorithm
    for alg in algorithms
        push!(cast_time_means, mean(df[df.algorithm .== alg, :cast_time]))
        push!(cast_time_stds, min(std(df[df.algorithm .== alg, :cast_time]), cast_time_means[end]))
        push!(alg_time_means, mean(df[df.algorithm .== alg, :calc_time]))
        push!(alg_time_stds, min(std(df[df.algorithm .== alg, :calc_time]), alg_time_means[end]))
        push!(alg_error_means, mean(df[df.algorithm .== alg, :error]))
        push!(alg_error_stds, min(std(df[df.algorithm .== alg, :error]), alg_error_means[end]))
        # Remove error bars if they go negative
        if cast_time_stds[length(cast_time_stds)] >= cast_time_means[length(cast_time_means)]
            cast_time_stds[length(cast_time_stds)] = 0
        end
        if alg_time_stds[length(alg_time_stds)] >= alg_time_means[length(alg_time_means)]
            alg_time_stds[length(alg_time_stds)] = 0
        end
        if alg_error_stds[length(alg_error_stds)] >= alg_error_means[length(alg_error_means)]
            alg_error_stds[length(alg_error_stds)] = 0
        end
    end
    max_x = maximum(alg_error_means+alg_error_stds)
    max_y = maximum(alg_time_means+alg_time_stds)
    min_y = minimum(alg_time_means-alg_time_stds)
    max_cast = maximum(cast_time_means+cast_time_stds)

    # If maximum error is 0, then all errors are the same. Set to arbitrary amount
    if (max_x == 0)
        max_x = 1
    end

    if isnan(max_x) || isnan(max_y)
        return
    end

    # Create axis
    ax1 = Axis(f[1, 1], yscale = log10, xlabel = "LU-based linear solution error", ylabel = "Calculation Time [s]",
                xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_label_size, yticklabelsize = tick_label_size,
                xtickformat = "{:.3e}", limits = (nothing, max_x*1.1, min_y*0.8, max_y*1.8))
    ax2 = Axis(f[1, 2], xlabel = "LU-based linear solution error", ylabel = "Casting Time [s]",
                xlabelsize = label_size, ylabelsize = label_size, xticklabelsize = tick_label_size, yticklabelsize = tick_label_size,
                xtickformat = "{:.3e}", ytickformat = "{:.3e}", limits = (nothing, max_x*1.1, -10^(-15)-1*max_cast*0.2, max_cast*1.2+10^(-15)))

    # Plot results
    scatter!(ax1, alg_error_means, alg_time_means, marker = shapes[1:length(alg_time_means)], markersize = tick_label_size, color = colors[1:length(alg_time_means)])
    errorbars!(ax1, alg_error_means, alg_time_means, alg_error_stds, color = colors[1:length(alg_time_means)], direction = :x, whiskerwidth = 10, linewidth = 3)
    errorbars!(ax1, alg_error_means, alg_time_means, alg_time_stds, color = colors[1:length(alg_time_means)], direction = :y, whiskerwidth = 10, linewidth = 3)
    foreach(i -> text!(ax1, position=(alg_error_means[i], alg_time_means[i]), fontsize=tick_label_size, rotation=0.5, align = (:left, :baseline), "   $(algorithms[i])"), 1:num_algs)

    scatter!(ax2, alg_error_means, cast_time_means, marker = shapes[1:length(cast_time_means)], markersize = tick_label_size, color = colors[1:length(cast_time_means)])
    errorbars!(ax2, alg_error_means, cast_time_means, alg_error_stds, color = colors[1:length(cast_time_means)], direction = :x, whiskerwidth = 10, linewidth = 3)
    errorbars!(ax2, alg_error_means, cast_time_means, cast_time_stds, color = colors[1:length(cast_time_means)], direction = :y, whiskerwidth = 10, linewidth = 3)
    foreach(i -> text!(ax2, position=(alg_error_means[i], cast_time_means[i]), fontsize=tick_label_size, rotation=0.5, align = (:left, :baseline), "   $(algorithms[i])"), 1:num_algs)

    # Create Title Label
    Label(f[1, 1:2, Top()], "$(uppercase(alg_name)) Decomposition Algorithm for $(mat_pattern): $(num_rows) x $(num_cols)", valign = :bottom, font = :bold, padding = (0, 0, 50, 0), fontsize = title_size)

    # Fix Margin/Spacing Issues

    Box(f[1, 1, Right()], width = 50, strokecolor = RGBAf(0.98, 0.98, 0.98, 0), color = RGBf(0.98, 0.98, 0.98))

    # Save figure
    mat_pattern = split(mat_pattern, "/")[end]
    save("$(alg_path)/$(alg_name)_$(mat_pattern)_$(num_rows) x $(num_cols).png", f)
end
