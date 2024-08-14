function plot_benchmark(alg_path, alg_name, df, ns, algs, mat_patterns, xaxis_type)
    algs_str = ["$a" for a in keys(algs)]
    for n in ns
        p = plot(
            [(
                ts = [
                       (
                           df′ = @views df[(df.pattern .== mat_pattern) .&&
                                           (df.n_cols .== n) .&& 
                                           (df.algorithm .== a), :];
                           if length(df′.time) > 0
                              minimum(df′.time)
                           else
                              0.0
                           end
                       )
                       for mat_pattern in reverse(mat_patterns)
                     ];
                 bar(name=a, x=ts, y=reverse(mat_patterns), orientation="h")
                ) for a in algs_str
             ])
        relayout!(p, barmode="group",
                     xaxis_type=xaxis_type,
                     xaxis_title="Time [s]",
                     yaxis_title="Matrix pattern, size $(n)x$(n)")
        savefig(p, "$(alg_path)/$(alg_name)_times_$(n)_$(xaxis_type).png", width=600, height=800, scale=1.5)
    end
end
