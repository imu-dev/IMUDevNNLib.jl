
"""
    skipfirstobs(x::AbstractArray)

Given an array `x` representing a time series, with the last dimension
corresponding to time, return a new array with the first observation removed.
"""
skipfirstobs(x::AbstractArray) = selectdim(x, ndims(x), 2:size(x, ndims(x)))

"""
    eachslicelastdim(m::AbstractArray{<:Any,N}) where {N}

Return an iterator over the last dimension of `m`.
"""
eachslicelastdim(m::AbstractArray{<:Any,N}) where {N} = eachslice(m; dims=N)
