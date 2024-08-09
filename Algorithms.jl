# Define wrappers to different LU algorithms and implementations
function dgetrf_a(A)
    if A isa Matrix
        t = @elapsed L, U, p = lu(A)
        err = norm(A[p,:] - L*U, 1)
    else # A isa SparseMatrixCSC
        t = @elapsed res = lu(A)
        b = rand(size(A,1))
        x = res \ b
        err = norm(A * x - b, 1)
    end
    return t, err
end
function umfpack_a(A)
    t = @elapsed res = lu(sparse(A))
    b = rand(size(A,1))
    x = res \ b
    err = norm(A * x - b, 1)
    return t, err
end
function klu_a(A)
    t = @elapsed K = klu(sparse(A))
    err = norm(K.L * K.U + K.F - K.Rs .\ A[K.p, K.q], 1)
    return t, err
end
function splu_a(A)
    t = @elapsed res = splu(sparse(A))
    b = rand(size(A,1))
    x = res \ b
    err = norm(A * x - b, 1) 
    return t, err
end

