"""
    TemporalData(xx::Vector{S}, yy::Vector{T}, x₀::S, y₀::T)

A type for temporal data, i.e. data that are indexed by time. Conceptually, `xx`
and `yy` are "inputs" and "targets" respectively. Both `xx` and `yy` are vectors
of arrays, where each array corresponds to a timepoint. Optionally, `x₀` and
`y₀` can be provided, which correspond to the initial state and the initial
observation (the latter is usually ignored).

    temporal_data([T::DataType], xx::AbstractArray, yy::AbstractArray; skipfirst=false)

Convenience constructor for `TemporalData` which accepts the data stored in
single arrays (as opposed to vector of arrays). It expects that the last
dimension of `xx` and `yy` corresponds to the time dimension, whereas the
penultimate dimension corresponds to the batch dimension. If `T` is provided,
it will be used to convert the data to the specified type. If `skipfirst` is
`true`, the first observation will be skipped for saving into internal `xx` and
`yy` and stored separately as `x₀` and `y₀` (it is convenient, as it is not
uncommon to treat the "zero'th" observation different from the rest).
"""
struct TemporalData{K,S<:AbstractArray{K},T<:AbstractArray{K}}
    xx::Vector{S}
    yy::Vector{T}
    x₀::S
    y₀::T
end

"""
Convenience method for creating a placeholder for the initial state when it is
not needed.
"""
function _empty_x₀(xx::AbstractArray)
    batch_size = size(xx)[end - 1]
    return zeros(eltype(xx), fill(0, ndims(xx) - 2)..., batch_size)
end

Base.eltype(::TemporalData{K}) where {K} = K

"""
    Flux.MLUtils.numobs(d::TemporalData)

For temporal data, the number of observations is synonymous with the batch size.
Batch size is going to be the last dimension of every array corresponding to
any timepoint.
"""
function Flux.MLUtils.numobs(d::TemporalData)
    isempty(d.yy) && return throw("TemporalData object contains no observations")
    return size(first(d.yy))[end]
end

function temporal_data(xx::AbstractArray, yy::AbstractArray; skipfirst=false)
    x₀, y₀ = if skipfirst
        xx = skipfirstobs(xx)
        yy = skipfirstobs(yy)
        selectfirstobs(xx), selectfirstobs(yy)
    else
        _empty_x₀(xx), _empty_x₀(yy)
    end
    return TemporalData([copy(x) for x in eachslicelastdim(xx)],
                        [copy(y) for y in eachslicelastdim(yy)],
                        copy(x₀),
                        copy(y₀))
end

function temporal_data(T::DataType, xx::AbstractArray, yy::AbstractArray;
                       skipfirst=false)
    return temporal_data(T.(xx), T.(yy); skipfirst)
end

"""
    Flux.MLUtils.getobs(d::TemporalData, i) 

Return the entire time series for the i-th observation (if `i` is a vector or a
range it will return a vector of time series for the correspoding batch.)
"""
function Flux.MLUtils.getobs(d::TemporalData, i)
    n_x = ndims(first(d.xx))
    n_y = ndims(first(d.yy))
    if isempty(d.x₀)
        return [selectdim(x, n_x, i) for x in d.xx], [selectdim(y, n_y, i) for y in d.yy]
    end
    return selectdim(d.x₀, n_x, i), selectdim(d.y₀, n_y, i),
           [selectdim(x, n_x, i) for x in d.xx], [selectdim(y, n_y, i) for y in d.yy]
end

num_samples(td::TemporalData) = size(first(td.xx))[end]
num_timepoints(td::TemporalData) = length(td.xx)
state_dim(td::TemporalData) = size(first(td.xx))[1:(end - 1)]
obs_dim(td::TemporalData) = size(first(td.yy))[1:(end - 1)]