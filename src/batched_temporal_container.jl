"""
    BatchedTemporalContainer(T::DataType,
                             size::NTuple{N,Int},
                             batchsize::Int) where {N}

A container that is used to write the results of operations performed on a
`TemporalData` object. Internally, it is an N-tensor (N > 2) where the last two
dimensions correspond to the batch and time dimensions respectively. Double
indexing into this object via `[i, t]` returns the (N-2)-dimensional array
corresponding to the `t`-th timepoint of the `i`-th batch. Single indexing
`[i]` returns the entire time series of the `i`-th batch.
"""
struct BatchedTemporalContainer{T<:AbstractArray}
    container::T
    batchsize::Int
    num_samples::Int
    num_batches::Int
    num_timepoints::Int

    function BatchedTemporalContainer(T::DataType, size::NTuple{N,Int},
                                      batchsize::Int) where {N}
        container = zeros(T, size...)
        num_samples, num_timepoints = size[(end - 1):end]
        num_batches = div(num_samples, batchsize) + (num_samples % batchsize != 0)
        return new{typeof(container)}(container, batchsize, num_samples, num_batches,
                                      num_timepoints)
    end
end

function Base.view(c::BatchedTemporalContainer, i::Int)
    if i > c.num_batches || i < 1
        throw(BoundsError(c, i))
    end

    indices = ((i - 1) * c.batchsize + 1):min(i * c.batchsize, c.num_samples)
    return selectdim(c.container, ndims(c.container) - 1, indices)
end

function Base.view(c::BatchedTemporalContainer, i::Int, j)
    temp = view(c, i)
    return selectdim(temp, ndims(temp), j)
end

function Base.getindex(c::BatchedTemporalContainer, i::Int)
    return copy(view(c, i))
end
Base.getindex(c::BatchedTemporalContainer, i, j) = copy(view(c, i, j))

function Base.setindex!(c::BatchedTemporalContainer, val, i)
    v = view(c, i)
    v .= val
    return v
end

function Base.setindex!(c::BatchedTemporalContainer, val, i, j)
    v = view(c, i, j)
    v .= val
    return v
end

function Base.firstindex(c::BatchedTemporalContainer, d=1)
    if d == 1
        return 1
    elseif d == 2
        return 1
    else
        throw(BoundsError(c, d))
    end
end

function Base.lastindex(c::BatchedTemporalContainer, d=1)
    if d == 1
        return c.num_batches
    elseif d == 2
        return c.num_timepoints
    else
        throw(BoundsError(c, d))
    end
end

"""
    getdata(c::BatchedTemporalContainer)

Return the underlying data of the container.
"""
getdata(c::BatchedTemporalContainer) = c.container
