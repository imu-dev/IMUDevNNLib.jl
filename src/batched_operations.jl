"""
    batched_matvecmul(A::Abstract3Tensor, B::AbstractMatrix)

NNlib's `batched_mul` may not behave as expected when one of the arguments is
a matrix. NNlib assumes that the matrix `B` must be broadcasted over each batch,
however, what might instead be desired is to treat `B` as a batch of vectors for
which a batch matrix-vector multiplication is desired.
"""
function batched_matvecmul(A::Abstract3Tensor, B::AbstractMatrix)
    m, num_batches = size(B)
    n = size(A, 1)
    C = reshape(B, m, 1, num_batches)
    return reshape(batched_mul(A, C), n, num_batches)
end

# -------------------------
# --- Inverse and solve ---
# -------------------------
#
# WARNING!
# --------
# Currently implemented batched_rinv is very limited, as we haven't yet reached
# a point when the time investment in a more general implementation encompassing
# CUDA would be justified. In the future we might put such an extension
# on a TODO list. Once we reach this point, we can revisit the following
# resources to get a good idea of how to implement it:
# https://github.com/JuliaGPU/CUDA.jl/blob/297c628756d25a55770ce50e99688a379cf77235/lib/cublas/wrappers.jl#L1662C4-L1662C16
# https://github.com/FluxML/NNlib.jl/blob/master/src/batched/batchedmul.jl
# and: https://github.com/pebeto/julia_extensions_example

"""
    batched_rinv(A::Abstract3Tensor)

Compute the inverse of each matrix in the batch `A`. Assumes that the last
dimension is the batch dimension.
"""
function batched_rinv(A::Abstract3Tensor)
    out = similar(A)
    for a in eachslice(A; dims=3)
        out[:, :, i] .= inv(a)
    end
    return out
end

"""
    batched_rinvsolve(A::Abstract3Tensor, B::Abstract3Tensor)

Compute the solution to the equation `A⁽ⁱ⁾X = B⁽ⁱ⁾` for each matrix pair
(A⁽ⁱ⁾, B⁽ⁱ⁾) in the batch. Assumes that the last dimension is the batch
dimension.
"""
function batched_rinvsolve(A::Abstract3Tensor, B::Abstract3Tensor)
    @assert size(A, 3) == size(B, 3) "batched matrix dimension must match"
    out = similar(A, typeof(inv(one(promote_type(eltype(A), eltype(B))))))
    return batched_rinvsolve!(out, A, B)
end

"""
    batched_rinvsolve!(out::Abstract3Tensor, A::Abstract3Tensor, B::Abstract3Tensor)

Compute the solution to the equation `A⁽ⁱ⁾X = B⁽ⁱ⁾` for each matrix pair
(A⁽ⁱ⁾, B⁽ⁱ⁾) in the batch and save the results to `out`. Assumes that the last
dimension is the batch dimension.
"""
function batched_rinvsolve!(out::Abstract3Tensor, A::Abstract3Tensor, B::Abstract3Tensor)
    for i in axes(out, 3)
        @views out[:, :, i] .= A[:, :, i] / B[:, :, i]
    end
    return out
end
