# Load AI-generated candidate algorithms
include("step1_candidate_algorithms_1d.jl")

# Connect candidate algorithms to SmartSolve
# Note: Commented-out algorithms were non-functional
SmartSolve.add_candidate_algorithm!(:fft1, recursive_fft)
#SmartSolve.add_candidate_algorithm!(:fft1, iterative_fft)
SmartSolve.add_candidate_algorithm!(:fft1, bluestein_fft)
#SmartSolve.add_candidate_algorithm!(:fft1, stockham_fft)
SmartSolve.add_candidate_algorithm!(:fft1, radix4_fft)
SmartSolve.add_candidate_algorithm!(:fft1, split_radix_fft)
#SmartSolve.add_candidate_algorithm!(:fft1, butterfly_fft)

# Benchmark all algorithms with an NxNxN tensor
SmartSolve.benchmark_algorithms!(:fft1, 2:10) do alg, n
    # Create a 1D input tensor of size (4^n)
    input = rand(ComplexF64, 4^n)
    # Benchmark this algorithm
    perf = @elapsed alg(input)
    # Return the performance result
    return perf
end