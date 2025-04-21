# Load SmartSolve
include("SmartSolve.jl")

# Load our database of algorithms
include("step2_connect_algorithms_1d.jl")
include("step2_connect_algorithms_2d.jl")

# Plot the performance results
p = SmartSolve.plot_benchmark_results(:fft1)
p = SmartSolve.plot_benchmark_results(:fft2)

# Select the best algorithm for a given input size
input = rand(ComplexF64, 4^1);
smartfft_1d! = SmartSolve.select_best_algorithm(:fft1, input)

input = rand(ComplexF64, 4^7);
smartfft_1d! = SmartSolve.select_best_algorithm(:fft1, input)

input = rand(ComplexF64, 4^1, 4^1);
smartfft_2d! = SmartSolve.select_best_algorithm(:fft2, input)

input = rand(ComplexF64, 4^5, 4^5);
smartfft_2d! = SmartSolve.select_best_algorithm(:fft2, input)