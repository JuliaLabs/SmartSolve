using FFTW

# 1. Radix-2 Cooley-Tukey Algorithm (Row-Column Method)
function fft2d_cooley_tukey!(A::Matrix{ComplexF64})
    N = size(A, 1)
    # Perform 1D FFT on each row
    for i in 1:N
        fft1d_cooley_tukey!(view(A, i, :))
    end
    # Perform 1D FFT on each column
    for j in 1:N
        fft1d_cooley_tukey!(view(A, :, j))
    end
    return A
end

# Helper 1D Cooley-Tukey FFT (recursive, in-place)
function fft1d_cooley_tukey!(x::AbstractVector{ComplexF64})
    N = length(x)
    if N ≤ 1 return x end
    
    # Check if N is a power of 2
    if N & (N - 1) != 0
        error("Length must be a power of 2")
    end
    
    # Split into even and odd indices
    even = x[1:2:N]
    odd = x[2:2:N]
    
    # Recursive calls
    fft1d_cooley_tukey!(even)
    fft1d_cooley_tukey!(odd)
    
    # Combine results
    for k in 1:N÷2
        t = exp(-2im * π * (k-1) / N) * odd[k]
        x[k] = even[k] + t
        x[k + N÷2] = even[k] - t
    end
    
    return x
end

# 2. Stockham Auto-Sort Algorithm
function fft2d_stockham!(A::Matrix{ComplexF64})
    N = size(A, 1)
    # Apply to rows
    for i in 1:N
        A[i, :] = fft1d_stockham(A[i, :])
    end
    # Apply to columns
    for j in 1:N
        A[:, j] = fft1d_stockham(A[:, j])
    end
    return A
end

function fft1d_stockham(x::AbstractVector{ComplexF64})
    N = length(x)
    if N ≤ 1 return copy(x) end
    
    # Check if N is a power of 2
    if N & (N - 1) != 0
        error("Length must be a power of 2")
    end
    
    # Two buffers to avoid allocation in the loop
    buffer1 = copy(x)
    buffer2 = similar(x)
    
    m = 1
    while m < N
        for k in 0:m-1
            w = exp(-2im * π * k / (2 * m))
            for j in 0:(N÷(2*m)-1)
                idx1 = j * 2 * m + k + 1
                idx2 = idx1 + m
                
                t1 = buffer1[idx1]
                t2 = buffer1[idx2] * w
                
                buffer2[j * m + k + 1] = t1 + t2
                buffer2[j * m + k + 1 + N÷2] = t1 - t2
            end
        end
        buffer1, buffer2 = buffer2, buffer1
        m *= 2
    end
    
    return buffer1
end

# 3. Eight-Step FFT Algorithm (Cache-Efficient)
function fft2d_eight_step!(A::Matrix{ComplexF64})
    N = size(A, 1)
    # 1. Transpose
    A = transpose!(copy(A), A)
    
    # 2. 1D FFTs on rows
    for i in 1:N
        A[i, :] = fft1d_iterative!(copy(A[i, :]))
    end
    
    # 3. Apply twiddle factors
    for j in 0:N-1
        for i in 0:N-1
            A[i+1, j+1] *= exp(-2im * π * i * j / N)
        end
    end
    
    # 4. Transpose
    A = transpose!(copy(A), A)
    
    # 5. 1D FFTs on rows (originally columns)
    for i in 1:N
        A[i, :] = fft1d_iterative!(copy(A[i, :]))
    end
    
    # 6. Transpose back to original orientation
    A = transpose!(copy(A), A)
    
    return A
end

# Helper iterative 1D FFT
function fft1d_iterative!(x::AbstractVector{ComplexF64})
    N = length(x)
    if N ≤ 1 return x end
    
    # Bit reversal
    j = 1
    for i in 1:N-1
        if i < j
            x[i], x[j] = x[j], x[i]
        end
        m = N ÷ 2
        while m ≥ 1 && j > m
            j -= m
            m ÷= 2
        end
        j += m
    end
    
    # Butterfly operations
    for s in 1:Int(log2(N))
        m = 2^s
        wm = exp(-2im * π / m)
        for k in 0:m:N-1
            w = 1.0 + 0.0im
            for j in 0:m÷2-1
                t = w * x[k + j + m÷2 + 1]
                u = x[k + j + 1]
                x[k + j + 1] = u + t
                x[k + j + m÷2 + 1] = u - t
                w *= wm
            end
        end
    end
    
    return x
end

# 4. Split-Radix FFT (Iterative Implementation)
function fft2d_split_radix!(A::Matrix{ComplexF64})
    N = size(A, 1)
    # Perform 1D FFT on each row
    for i in 1:N
        fft1d_split_radix!(view(A, i, :))
    end
    # Perform 1D FFT on each column
    for j in 1:N
        fft1d_split_radix!(view(A, :, j))
    end
    return A
end

function fft1d_split_radix!(x::AbstractVector{ComplexF64})
    N = length(x)
    if N <= 1 return x end
    
    # Check if N is a power of 2
    if N & (N - 1) != 0
        error("Length must be a power of 2")
    end
    
    # Bit-reversal permutation (similar to Cooley-Tukey)
    j = 1
    for i in 1:N-1
        if i < j
            x[i], x[j] = x[j], x[i]
        end
        m = N ÷ 2
        while m ≥ 1 && j > m
            j -= m
            m ÷= 2
        end
        j += m
    end
    
    # L-shaped butterflies for split-radix
    L = 2
    while L <= N
        # Radix-2 part
        m = L ÷ 2
        wm = exp(-2im * π / L)
        
        for k in 0:L:N-1
            w = 1.0 + 0.0im
            for j in 0:m-1
                temp = w * x[k+j+m+1]
                x[k+j+m+1] = x[k+j+1] - temp
                x[k+j+1] += temp
                w *= wm
            end
        end
        
        # Extra butterflies for split-radix strategy when L >= 4
        if L >= 4
            m = L ÷ 4
            wm = exp(-2im * π / L)
            for k in L÷2:L:N-1
                w = exp(-2im * π * (k % L) / L)
                for j in 0:m-1
                    idx1 = k + j + 1
                    idx2 = k + j + m + 1
                    
                    temp1 = x[idx1]
                    temp2 = x[idx2]
                    
                    x[idx1] = (temp1 + 1im * temp2) / sqrt(2.0)
                    x[idx2] = (temp1 - 1im * temp2) / sqrt(2.0)
                end
            end
        end
        
        L *= 2
    end
    
    return x
end

# 5. Bluestein's FFT Algorithm (Chirp Z-Transform)
function fft2d_bluestein!(A::Matrix{ComplexF64})
    N = size(A, 1)
    # Perform 1D FFT on each row
    for i in 1:N
        A[i, :] = fft1d_bluestein!(copy(A[i, :]))
    end
    # Perform 1D FFT on each column
    for j in 1:N
        A[:, j] = fft1d_bluestein!(copy(A[:, j]))
    end
    return A
end

function fft1d_bluestein!(x::AbstractVector{ComplexF64})
    N = length(x)
    if N ≤ 1 return x end
    
    # Find M that is a power of 2 and M ≥ 2N-1
    M = 1
    while M < 2*N-1
        M *= 2
    end
    
    # Precompute chirp factors with explicit handling of indices
    a = [exp(im * π * (n^2) / N) for n in 0:N-1]
    
    # Create b with careful indexing to avoid out-of-bounds access
    b = zeros(ComplexF64, 2*N-1)
    for n in -(N-1):(N-1)
        b[n+(N-1)+1] = exp(-im * π * (n^2) / N)
    end
    
    # Zero pad sequences to length M
    a_padded = zeros(ComplexF64, M)
    b_padded = zeros(ComplexF64, M)
    
    # Fill a_padded with data * chirp
    for n in 1:N
        a_padded[n] = x[n] * a[n]
    end
    
    # Fill b_padded with proper indexing
    for n in 1:(2*N-1)
        b_padded[n] = b[n]
    end
    
    # Perform convolution using FFT
    A_fft = fft1d_iterative!(copy(a_padded))
    B_fft = fft1d_iterative!(copy(b_padded))
    
    # Element-wise multiplication
    C_fft = A_fft .* B_fft
    
    # Inverse FFT
    c = fft1d_iterative!(copy(C_fft))
    c ./= M  # Normalization for inverse FFT
    
    # Extract result and apply chirp correction safely
    result = similar(x)
    for n in 1:N
        result[n] = c[n] * conj(a[n])
    end
    
    return result
end

# Helper function for transposes
function transpose!(dest::Matrix{ComplexF64}, src::Matrix{ComplexF64})
    n, m = size(src)
    for j in 1:m
        for i in 1:n
            dest[j, i] = src[i, j]
        end
    end
    return dest
end

# Example usage:
using FFTW
function test_fft(N::Int=8)
    # Create test matrix
    A = [ComplexF64(i + j) for i in 1:N, j in 1:N]
    A_copy = copy(A)
    
    # Test implementations
    println("Testing Cooley-Tukey implementation:")
    result1 = fft2d_cooley_tukey!(copy(A))
    
    println("Testing Stockham implementation:")
    result2 = fft2d_stockham!(copy(A))
    
    println("Testing Eight-Step implementation:")
    result3 = fft2d_eight_step!(copy(A))
    
    println("Testing Split-Radix implementation:")
    result4 = fft2d_split_radix!(copy(A))
    
    println("Testing Bluestein implementation:")
    result5 = fft2d_bluestein!(copy(A))
    
    # Verify results against Julia's built-in FFT
    reference = fft(A_copy)
    
    println("\nMaximum difference from reference:")
    println("Cooley-Tukey: ", maximum(abs.(result1 - reference)))
    println("Stockham: ", maximum(abs.(result2 - reference)))
    println("Eight-Step: ", maximum(abs.(result3 - reference)))
    println("Split-Radix: ", maximum(abs.(result4 - reference)))
    println("Bluestein: ", maximum(abs.(result5 - reference)))
end