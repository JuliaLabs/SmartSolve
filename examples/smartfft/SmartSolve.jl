module SmartSolve

using Plots

const ALGORITHMS = Dict{Symbol, Vector{Function}}()

function add_candidate_algorithm!(alg_name::Symbol, alg_func::Function)
    if !haskey(ALGORITHMS, alg_name)
        ALGORITHMS[alg_name] = Function[]
    end
    push!(ALGORITHMS[alg_name], alg_func)
end

const BENCHMARK_RESULTS = Dict{Symbol, Dict{Function, Dict{Int, Vector{Float64}}}}()

function benchmark_algorithms!(f::Function, alg_name::Symbol, Ns)
    if !haskey(BENCHMARK_RESULTS, alg_name)
        BENCHMARK_RESULTS[alg_name] = Dict{Function, Dict{Int, Vector{Float64}}}()
    end
    for alg in ALGORITHMS[alg_name]
        @info "Benchmarking $alg_name with $alg"
        if !haskey(BENCHMARK_RESULTS[alg_name], alg)
            BENCHMARK_RESULTS[alg_name][alg] = Dict{Int, Vector{Float64}}()
        end
        for n in Ns
            if !haskey(BENCHMARK_RESULTS[alg_name][alg], n)
                BENCHMARK_RESULTS[alg_name][alg][n] = Float64[]
            end
            times = Float64[]
            for iter in 1:10
                t = f(alg, n)
                push!(times, t)
            end
            BENCHMARK_RESULTS[alg_name][alg][n] = times
        end
    end
end

function select_best_algorithm(alg_name::Symbol, input::Array)
    if !haskey(BENCHMARK_RESULTS, alg_name)
        error("No benchmark results for $alg_name")
    end
    best_alg = nothing
    best_time = Inf
    first_alg = first(ALGORITHMS[alg_name])
    Ns = sort(collect(keys(BENCHMARK_RESULTS[alg_name][first_alg])))
    closest_n = Ns[argmin(abs.(Ns .- size(input, 1)))]
    for alg in ALGORITHMS[alg_name]
        times = BENCHMARK_RESULTS[alg_name][alg][closest_n]
        time = minimum(times)
        if time < best_time
            best_time = time
            best_alg = alg
        end
    end
    return best_alg
end

function benchmark_results(alg_name::Symbol)
    if !haskey(BENCHMARK_RESULTS, alg_name)
        error("No benchmark results for $alg_name")
    end
    first_alg = first(ALGORITHMS[alg_name])
    results = zeros(length(ALGORITHMS[alg_name]), length(BENCHMARK_RESULTS[alg_name][first_alg]))
    i = 0
    for alg in ALGORITHMS[alg_name]
        i += 1
        j = 0
        for n in sort(collect(keys(BENCHMARK_RESULTS[alg_name][alg])))
            j += 1
            times = BENCHMARK_RESULTS[alg_name][alg][n]
            results[i, j] = minimum(times)
        end
    end
    return results
end

function plot_benchmark_results(alg_name::Symbol)
    results = benchmark_results(alg_name)
    algs = permutedims(map(string, collect(ALGORITHMS[alg_name])))
    return plot(results', yscale=:log10,
                title="FFT Performance", xlabel="Input Size", ylabel="Time (s)",
                legend=:outertopright,
                labels=algs)
end

end # module SmartSolve