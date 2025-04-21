using FFTW

### 1. Recursive Radix-2 FFT (Cooley-Tukey Algorithm)
function recursive_fft(x::Vector{<:Number})
    n = length(x)
    if n == 1
        return x
    end
    
    # Split into even and odd indices
    even = recursive_fft(x[1:2:n])
    odd = recursive_fft(x[2:2:n])
    
    # Combine results
    result = zeros(Complex{Float64}, n)
    for k in 0:n÷2-1
        t = exp(-2im * π * k / n) * odd[k+1]
        result[k+1] = even[k+1] + t
        result[k+1+n÷2] = even[k+1] - t
    end
    
    return result
end

### 2. Iterative Radix-2 FFT (Cooley-Tukey Algorithm)
function iterative_fft(x::Vector{<:Number})
    n = length(x)
    
    # Bit-reversal permutation
    result = complex.(x)
    j = 1
    for i in 1:n-1
        if i < j
            result[i], result[j] = result[j], result[i]
        end
        m = n ÷ 2
        while m <= j
            j -= m
            m ÷= 2
        end
        j += m
    end
    
    # Butterflies
    for s in 1:log2(n)
        m = 2^s
        wm = exp(-2im * π / m)
        for k in 0:m:n-1
            w = 1.0 + 0.0im
            for j in 0:m÷2-1
                t = w * result[k+j+m÷2+1]
                u = result[k+j+1]
                result[k+j+1] = u + t
                result[k+j+m÷2+1] = u - t
                w *= wm
            end
        end
    end
    
    return result
end

### 3. Bluestein's FFT Algorithm (Chirp Z-Transform) - Handles any size n
function bluestein_fft(x::Vector{<:Number})
    n = length(x)
    
    # Find next power of 2 >= 2n-1
    m = 2^ceil(Int, log2(2n-1))
    
    # Create chirp sequences
    a = [exp(im * π * (k^2) / n) for k in 0:n-1]
    b = [exp(-im * π * (k^2) / n) for k in -(n-1):n-1]
    
    # Zero padding
    a_padded = zeros(Complex{Float64}, m)
    b_padded = zeros(Complex{Float64}, m)
    
    # Fill in values
    a_padded[1:n] = x .* a
    b_padded[1:2n-1] = b
    
    # Convolve using FFT (assuming the presence of an FFT function)
    c_padded = ifft(fft(a_padded) .* fft(b_padded))[1:n]
    
    # Adjust final result
    result = c_padded .* a
    
    return result
end

### 4. Stockham Radix-2 FFT Algorithm (Auto-Sorting)
function stockham_fft(x::Vector{<:Number})
    n = length(x)
    
    # Ensure n is a power of 2
    if log2(n) != floor(log2(n))
        error("Input length must be a power of 2")
    end
    
    # Initialize buffers
    buffer1 = complex.(x)
    buffer2 = zeros(Complex{Float64}, n)
    
    for s in 1:Int(log2(n))
        m = 2^s
        half_m = m ÷ 2
        
        for k in 0:half_m:n-1
            w = 1.0 + 0.0im
            wm = exp(-2im * π / m)
            
            for j in 0:half_m-1
                even_idx = k + j + 1
                odd_idx = even_idx + half_m
                
                # Butterfly operation
                even_val = buffer1[even_idx]
                odd_val = buffer1[odd_idx]
                
                buffer2[k÷half_m * m + j + 1] = even_val + w * odd_val
                buffer2[k÷half_m * m + j + half_m + 1] = even_val - w * odd_val
                
                w *= wm
            end
        end
        
        # Swap buffers for next iteration
        buffer1, buffer2 = buffer2, buffer1
    end
    
    return buffer1
end

### 5. Radix-4 FFT Algorithm
function radix4_fft(x::Vector{<:Number})
    n = length(x)
    
    # Ensure n is a power of 4
    if log2(n) % 2 != 0
        error("Input length must be a power of 4")
    end
    
    if n == 1
        return x
    end
    
    # Split into four parts
    x0 = radix4_fft(x[1:4:n])
    x1 = radix4_fft(x[2:4:n])
    x2 = radix4_fft(x[3:4:n])
    x3 = radix4_fft(x[4:4:n])
    
    # Combine results
    result = zeros(Complex{Float64}, n)
    for k in 0:n÷4-1
        w1 = exp(-2im * π * k / n)
        w2 = w1 * w1
        w3 = w2 * w1
        
        t0 = x0[k+1]
        t1 = w1 * x1[k+1]
        t2 = w2 * x2[k+1]
        t3 = w3 * x3[k+1]
        
        result[k+1] = t0 + t1 + t2 + t3
        result[k+1+n÷4] = t0 - im*t1 - t2 + im*t3
        result[k+1+2n÷4] = t0 - t1 + t2 - t3
        result[k+1+3n÷4] = t0 + im*t1 - t2 - im*t3
    end
    
    return result
end

### 6. Split-Radix FFT Algorithm
function split_radix_fft(x::Vector{<:Number})
    n = length(x)
    
    if n == 1
        return x
    end
    
    if n == 2
        return [x[1] + x[2], x[1] - x[2]]
    end
    
    # Split into even, every 4th, and every 4th+2
    even = split_radix_fft(x[1:2:n])
    odd1 = split_radix_fft(x[2:4:n])
    odd3 = split_radix_fft(x[4:4:n])
    
    # Combine results
    result = zeros(Complex{Float64}, n)
    for k in 0:n÷4-1
        w1 = exp(-2im * π * k / n)
        w3 = exp(-6im * π * k / n)
        
        result[k+1] = even[k+1] + w1 * odd1[k+1] + w3 * odd3[k+1]
        result[k+1+n÷2] = even[k+1] - w1 * odd1[k+1] - w3 * odd3[k+1]
        
        result[k+1+n÷4] = even[k+1+n÷4] - im * (w1 * odd1[k+1] - w3 * odd3[k+1])
        result[k+1+3n÷4] = even[k+1+n÷4] + im * (w1 * odd1[k+1] - w3 * odd3[k+1])
    end
    
    return result
end

### 7. Prime-Factor Algorithm (PFA) for FFT
function prime_factor_fft(x::Vector{<:Number}, n1::Int, n2::Int)
    n = length(x)
    
    # Ensure n = n1 * n2 and n1, n2 are coprime
    if n != n1 * n2
        error("n must equal n1 * n2")
    end
    
    # Create 2D array from 1D
    X = zeros(Complex{Float64}, n1, n2)
    for i in 0:n-1
        i1 = i % n1
        i2 = i % n2
        X[i1+1, i2+1] = x[i+1]
    end
    
    # FFT along each dimension
    for i1 in 1:n1
        X[i1, :] = recursive_fft(X[i1, :])
    end
    
    for i2 in 1:n2
        X[:, i2] = recursive_fft(X[:, i2])
    end
    
    # Map back to 1D
    result = zeros(Complex{Float64}, n)
    for i1 in 0:n1-1
        for i2 in 0:n2-1
            k = (i1 * n2 + i2) % n
            result[k+1] = X[i1+1, i2+1]
        end
    end
    
    return result
end

### 8. FFT using Two Butterfly Algorithms (Radix-2^2)
function butterfly_fft(x::Vector{<:Number})
    n = length(x)
    
    # Bit-reversal permutation
    result = complex.(x)
    j = 1
    for i in 1:n-1
        if i < j
            result[i], result[j] = result[j], result[i]
        end
        m = n ÷ 2
        while m <= j
            j -= m
            m ÷= 2
        end
        j += m
    end
    
    # 2-point DFT (butterfly)
    for i in 1:2:n
        t = result[i]
        result[i] = t + result[i+1]
        result[i+1] = t - result[i+1]
    end
    
    # 4-point and higher radix butterflies
    for stage in 2:log2(n)
        m = 2^stage
        half_m = m ÷ 2
        
        for k in 0:m:n-1
            for j in 0:half_m-1
                idx1 = k + j + 1
                idx2 = idx1 + half_m
                w = exp(-2im * π * j / m)
                t = w * result[idx2]
                result[idx2] = result[idx1] - t
                result[idx1] = result[idx1] + t
            end
        end
    end
    
    return result
end