# Load our optimizing scheduler, Dagger
using Dagger

# Load our benchmarked database of algorithms
include("step3_selection_best_algorithm.jl")

# Let's compare against serial FFTW
using FFTW
input = rand(ComplexF64, 4^5, 4^5);
println("FFTW")
FFTW.set_num_threads(1)
@time fft!(input);
println("SmartSolve")
@time smartfft_2d!(input);

# If our data is large, Dagger can scale the algorithm for us
include("generate_gantt.jl")

# TODO: This will use SmartSolve's FFT within Dagger's Pencil FFT,
# potentially even allowing each pencil's FFT to be optimized for
# where the pencil will execute (optimize for specific CPU/GPU architecture)
function daggerfft!(A, B)
    # This is a fully parallel 2D FFT algorithm
    # It supports multi-CPU, multi-GPU, and distributed computing
    A_parts = A.chunks
    B_parts = B.chunks
    Dagger.spawn_datadeps() do
        for idx in eachindex(A_parts)
            Dagger.@spawn name="smartfft!(dim 1)" smartfft!(InOut(A_parts[idx]))
        end
    end
    copyto!(B, A)
    Dagger.spawn_datadeps() do
        for idx in eachindex(B_parts)
            Dagger.@spawn name="smartfft!(dim 2)" smartfft!(InOut(B_parts[idx]))
        end
    end
end
A = rand(Blocks(4^5, div(4^5, 16)), ComplexF64, 4^5, 4^5);
B = zeros(Blocks(div(4^5, 16), 4^5), ComplexF64, 4^5, 4^5);
@time daggerfft!(A, B);