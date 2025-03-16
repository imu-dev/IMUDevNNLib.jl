"""
    $(TYPEDEF)

An array `data` that is split into blocks of batches of size `batchsize` each
(with the last batch possibly shrunk to fit the `data`'s dimensions). Indexing
effectively works like indexing over a `Vector` of subarrays, where each
subarray corresponds to a batch.

# Constructors

    batchdelimitedarray(K::DataType, size::NTuple{<:Any,Int}, batchsize::Int)

Create a `BatchDelimitedArray` with underlying `data` of `size` and eltype `K`.

    batchdelimitedarray(data::AbstractArray, batchsize::Int)

Same as the default `BatchDelimitedArray`'s constructor.

!!! tip
    Using a constructor that expects `data::AbstractArray` can be handy if you
    wish to use a `Mmap`-ed array.
"""
struct BatchDelimitedArray{T,S<:AbstractArray} <: AbstractArray{T,1}
    data::S
    batchsize::Int
    num_samples::Int
    num_batches::Int

    function BatchDelimitedArray(data::AbstractArray,
                                 batchsize::Int)
        num_samples = size(data, ndims(data))
        num_batches = div(num_samples, batchsize) + (num_samples % batchsize != 0)
        T = typeof(zeros(K, size(data)[1:(end - 1)]...))
        return new{T,typeof(data)}(data, batchsize, num_samples, num_batches)
    end
end

function batchdelimitedarray(K::DataType,
                             size::NTuple{<:Any,Int},
                             batchsize::Int)
    data = zeros(K, size...)
    return BatchDelimitedArray(data, batchsize)
end

function batchdelimitedarray(data::AbstractArray,
                             batchsize::Int)
    return BatchDelimitedArray(data, batchsize)
end

function Base.view(a::BatchDelimitedArray, i::Int)
    if i > a.num_batches || i < 1
        throw(BoundsError(a, i))
    end

    indices = ((i - 1) * a.batchsize + 1):min(i * a.batchsize, a.num_samples)
    return selectalonglastdim(a.data, indices)
end

Base.getindex(a::BatchDelimitedArray, i::Int) = copy(view(a, i))

function Base.setindex!(a::BatchDelimitedArray, val, i)
    v = view(a, i)
    v .= val
    return v
end

function Base.firstindex(a::BatchDelimitedArray, d=1)
    if d == 1
        return 1
    else
        throw(BoundsError(a, d))
    end
end

function Base.lastindex(a::BatchDelimitedArray, d=1)
    if a == 1
        return a.num_batches
    else
        throw(BoundsError(a, d))
    end
end

"""
    getdata(a::BatchDelimitedArray)

Return the underlying data of the container.
"""
getdata(a::BatchDelimitedArray) = a.data