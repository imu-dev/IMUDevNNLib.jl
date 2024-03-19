"""
    skipfirstobs(x::AbstractArray; timedim=ndims(x)-1)

Given an array `x` representing a time series, with the `timedim` dimension
corresponding to time, return a new array with the first observation removed.
"""
function skipfirstobs(x::AbstractArray; timedim=ndims(x) - 1)
    return selectdim(x, timedim, 2:size(x, timedim))
end

"""
    selectfirstobs(x::AbstractArray; timedim=ndims(x)-1)

Given an array `x` representing a time series, with the `timedim` dimension
corresponding to time, return a new array with only the first observation.
"""
function selectfirstobs(x::AbstractArray; timedim=ndims(x) - 1)
    return selectdim(x, timedim, 1)
end

"""
    peelfirstobs(x::AbstractArray; timedim=ndims(x)-1)

Given an array `x` representing a time series, with the `timedim` dimension
corresponding to time, return two array views:
- view into the first observation,
- view into the remaining observations.
"""
function peelfirstobs(x::AbstractArray; timedim=ndims(x) - 1)
    return selectfirstobs(x; timedim), skipfirstobs(x; timedim)
end

"""
    selectalonglastdim(m::AbstractArray, i)

Select subview of the array `m` with index `i` accessed along the last dimension
of this array.
"""
selectalonglastdim(m::AbstractArray{<:Any,N}, i) where {N} = selectdim(m, N, i)

"""
    mergealonglastdim(data::Vector{<:AbstractArray})

Merge a collection of arrays along their last dimension. The arrays must agree
in all dimensions except the last one.

!!! tip
    yes, `cat(data...; dims=ndims(first(data)))` will work, but for large arrays
    it will be very, very inefficient, in which case you're better off using
    `mergealonglastdim`.
"""
function mergealonglastdim(data::Vector{<:AbstractArray})
    length(unique(map(d -> size(d)[1:(ndims(d) - 1)], data))) == 1 ||
        throw("elements of a collection must agree in all must the last dimension")
    lastdim_sizes = map(d -> size(d, ndims(d)), data)
    out = similar(first(data), size(first(data))[1:(end - 1)]..., sum(lastdim_sizes))
    indices = vcat(1, cumsum(lastdim_sizes) .+ 1)[1:(end - 1)]
    for (i, s, d) in zip(indices, lastdim_sizes, data)
        selectalonglastdim(out, i:(i + s - 1)) .= d
    end
    return out
end

"""
    nnflatten(x::AbstractArray{<:Any,N}) where {N}

A common flattening scheme encountered in machine learning libraries: flatten
the array `x` into a 2D array by squashing all dimensions except the last one
(i.e. except the batch dimension).
"""
nnflatten(x::AbstractArray{<:Any,N}) where {N} = reshape(x, :, size(x, N))