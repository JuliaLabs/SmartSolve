# Load AI-generated candidate algorithms
include("step1_candidate_algorithms_2d.jl")

# Connect candidate algorithms to SmartSolve
SmartSolve.add_candidate_algorithm!(:fft2, fft2d_cooley_tukey!)
SmartSolve.add_candidate_algorithm!(:fft2, fft2d_stockham!)
SmartSolve.add_candidate_algorithm!(:fft2, fft2d_eight_step!)
SmartSolve.add_candidate_algorithm!(:fft2, fft2d_split_radix!)
SmartSolve.add_candidate_algorithm!(:fft2, fft2d_bluestein!)

# Benchmark all algorithms with an NxNxN tensor
SmartSolve.benchmark_algorithms!(:fft2, 2:5) do alg, n
    # Create a 1D input tensor of size (4^n, 4^n)
    input = rand(ComplexF64, 4^n, 4^n)
    # Benchmark this algorithm
    perf = @elapsed alg(input)
    # Return the performance result
    return perf
end