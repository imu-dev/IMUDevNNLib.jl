"""
    TemporalData(x::Vector{S}, y::Vector{T})

A type for temporal data, i.e. data that are indexed by time. Conceptually, `x`
and `y` are "inputs" and "targets" respectively. Both `x` and `y` are vectors of
arrays, where each array corresponds to a timepoint.

    temporal_data([T::DataType], xx::AbstractArray, yy::AbstractArray; skipfirst=false)

Convenience constructor for `TemporalData` which accepts the data stored in
single arrays (as opposed to vector of arrays). It expects that the last
dimension of `xx` and `yy` corresponds to the time dimension, whereas the
penultimate dimension corresponds to the batch dimension. If `T` is provided,
it will be used to convert the data to the specified type. If `skipfirst` is
`true`, the first observation will be skipped (it is convenient, as it is not
uncommon to treat the "zero'th" observation different from the rest).
"""
struct TemporalData{K,S<:AbstractArray{K},T<:AbstractArray{K}}
    x::Vector{S}
    y::Vector{T}
end

Base.eltype(::TemporalData{K}) where {K} = K

"""
    MLUtils.numobs(d::TemporalData)

For temporal data, the number of observations is synonymous with the batch size.
Batch size is going to be the last dimension of every array corresponding to
any timepoint.
"""
function MLUtils.numobs(d::TemporalData)
    isempty(d.y) && return throw("No observations in the data")
    return size(first(d.y))[end]
end

function temporal_data(xx::AbstractArray, yy::AbstractArray; skipfirst=false)
    if skipfirst
        xx = skipfirstobs(xx)
        yy = skipfirstobs(yy)
    end
    return TemporalData([copy(x) for x in eachslicelastdim(xx)],
                        [copy(y) for y in eachslicelastdim(yy)])
end

function temporal_data(T::DataType, xx::AbstractArray, yy::AbstractArray; skipfirst=false)
    if skipfirst
        xx = skipfirstobs(xx)
        yy = skipfirstobs(yy)
    end
    return TemporalData([copy(x) for x in eachslicelastdim(T.(xx))],
                        [copy(y) for y in eachslicelastdim(T.(yy))])
end

"""
    MLUtils.getobs(d::TemporalData, i) 

Return the entire time series for the i-th observation (if `i` is a vector or a
range it will return a vector of time series for the correspoding batch.)
"""
function MLUtils.getobs(d::TemporalData, i)
    n_x = ndims(first(d.x))
    n_y = ndims(first(d.y))
    return [selectdim(x, n_x, i) for x in d.x], [selectdim(y, n_y, i) for y in d.y]
end
