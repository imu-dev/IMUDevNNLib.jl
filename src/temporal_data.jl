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

is_skipfirst(td::TemporalData) = !isempty(td.x₀) || !isempty(td.y₀)

"""
    MLUtils.numobs(d::TemporalData)

For temporal data, the number of observations is synonymous with the batch size.
Batch size is going to be the last dimension of every array corresponding to
any timepoint.
"""
function MLUtils.numobs(d::TemporalData)
    isempty(d.yy) && return throw("TemporalData object contains no observations")
    return size(first(d.yy))[end]
end

function temporal_data(xx::AbstractArray, yy::AbstractArray;
                       skipfirst=false,
                       x₀::Union{<:AbstractArray,Missing}=missing,
                       y₀::Union{<:AbstractArray,Missing}=missing,)
    _x₀, _y₀ = if skipfirst
        xx = skipfirstobs(xx)
        yy = skipfirstobs(yy)
        selectfirstobs(xx), selectfirstobs(yy)
    else
        _empty_x₀(xx), _empty_x₀(yy)
    end
    if ismissing(x₀)
        x₀ = _x₀
    end
    if ismissing(y₀)
        y₀ = _y₀
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

function temporal_data(; xx::Union{<:AbstractArray,Missing}=missing,
                       yy::Union{<:AbstractArray,Missing}=missing,
                       x₀::Union{<:AbstractArray,Missing}=missing,
                       y₀::Union{<:AbstractArray,Missing}=missing,
                       skipfirst=false)
    ismissing(xx) && ismissing(yy) && throw("At least one of `xx` or `yy` must be provided")
    if ismissing(xx)
        xx = similar(yy, 0, size(yy)[(end - 1):end]...)
    end
    if ismissing(yy)
        yy = similar(xx, 0, size(xx)[(end - 1):end]...)
    end
    return temporal_data(xx, yy; skipfirst, x₀, y₀)
end

function temporal_data(T::DataType;
                       xx::Union{<:AbstractArray,Missing}=missing,
                       yy::Union{<:AbstractArray,Missing}=missing,
                       x₀::Union{<:AbstractArray,Missing}=missing,
                       y₀::Union{<:AbstractArray,Missing}=missing,
                       skipfirst=false)
    return temporal_data(; xx=ismissing(xx) ? xx : T.(xx),
                         yy=ismissing(yy) ? yy : T.(yy),
                         x₀=ismissing(x₀) ? x₀ : T.(x₀),
                         y₀=ismissing(y₀) ? y₀ : T.(y₀),
                         skipfirst)
end

"""
    MLUtils.getobs(d::TemporalData, i) 

Return the entire time series for the i-th observation (if `i` is a vector or a
range it will return a vector of time series for the correspoding batch.)
"""
function MLUtils.getobs(d::TemporalData, i)
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
function state_dim(td::TemporalData)
    x = isempty(first(td.xx)) ? td.x₀ : first(td.xx)
    isempty(x) && throw("State dimension cannot be determined")
    return size(x)[1:(end - 1)]
end
function obs_dim(td::TemporalData)
    y = isempty(first(td.yy)) ? td.y₀ : first(td.yy)
    isempty(y) && throw("Observation dimension cannot be determined")
    return size(y)[1:(end - 1)]
end

"""
Return the size of the output container for the given `TemporalData` object.
This can be either:
- the size of the state (if `of_what` is `:state` or `:xx`)
- the size of the observation (if `of_what` is `:obs`, `:observation` or `:yy`)
Note that the output container is assumed to be an N-tensor (N > 2) where the
last two dimensions correspond to the batch and time dimensions respectively;
and in particular, **the time dimension includes the initial state!**
"""
Base.size(td::TemporalData, of_what::Symbol; kwargs...) = size(td, Val(of_what); kwargs...)

function Base.size(td::TemporalData, ::Union{Val{:state},Val{:xx}}; state_size=nothing)
    state_size = try
        state_dim(td)
    catch e
        isnothing(state_size) && throw(e)
    end
    return (state_size..., num_samples(td), num_timepoints(td) + is_skipfirst(td))
end

function Base.size(td::TemporalData,
                   ::Union{Val{:obs},Val{:observation},Val{:observations},Val{:yy}};
                   obs_size=nothing)
    obs_size = try
        obs_dim(td)
    catch e
        isnothing(obs_size) && throw(e)
    end
    return (obs_size..., num_samples(td), num_timepoints(td) + is_skipfirst(td))
end

function out_container(td::TemporalData, of_what::Symbol; batchsize, kwargs...)
    return BatchedTemporalContainer(eltype(td),
                                    size(td, Val(of_what); kwargs...),
                                    batchsize)
end