"""
    $(TYPEDEF)

Parent to all defined data layouts. First and foremost, `DataLayout`s are merely
conventions for consistently arranging data for analysis. From a more practical
standpoint however, there are a couple of `Base.reshape` implementations that
allow for transforming between different layouts. Additionally, there are also
some utility functions.
"""
abstract type DataLayout end

Base.Broadcast.broadcastable(layout::DataLayout) = Ref(layout)

"""
    $(TYPEDEF)

Data stored in `SingleTimeseriesLayout` comprises of a single long timeseries
stored in a single array. It's a (N-1)-dimensional array with
- the first (N-2)-dimensions corresponding to the state space
- the last (N-1)-th dimension corresponding to time

!!! note
    There is only one sample, so one dimension that's usually reserved for
    samples (i.e. N-th dimension) is dropped.

!!! tip "Example"
    Consider a single recording from an IMU device that's been recording for one
    day @1Hz. The data extracted from this device in a `SingleTimeseriesLayout`
    would be represented as a `Matrix` with dimensions `3×86_400`, with each
    column representing a single observation of the 3d-acceleration.
"""
struct SingleTimeseriesLayout <: DataLayout end

"""
    $(TYPEDEF)

Data stored in `TimeseriesLayout` comprises of a collection (usually a vector)
of arrays, which conceptually represents a timeseries of observations for
multiple samples. Each element of the collection is an (N-1)-dimensional array
with:
- the first (N-2)-dimensions corresponding to the state space
- the last (N-1)-th dimension corresponding to samples

The collection indexes over timepoints (which can be thought of as the N-th
dimension).

!!! tip "Example"
    Consider two recordings from two synchronized IMU devices that have been
    recording for one day @1Hz. The data extracted from these two devices in a
    `TimeseriesLayout` would be represented as a `Vector{Matrix}` with length
    `86_400`, where each element of this vector would have dimensions `3×2`
    (i.e. it would be two vectors of 3d-accelerations horizontally concatenated
    together into a `3×2` `Matrix`).
"""
struct TimeseriesLayout <: DataLayout end

"""
    $(TYPEDEF)

Data stored as `SingleObsTimeseriesLayout` is simply a single array element
(conceptually an observation) from the collection that's stored in
[`TimeseriesLayout`](@ref).
"""
struct SingleObsTimeseriesLayout <: DataLayout end

"""
    $(TYPEDEF)

Data stored in the `StackedArrayLayout` is an N-dimensional array with the
last, N-th dimension indexing over samples.

!!! note
    For a particular case of `StackedArrayLayout` intended to respresent time
    series data we follow the convention of using an N-dimensional array with:
    - the first (N-2)-dimensions corresponding to the state space
    - the (N-1)-th dimension corresponding to time, and
    - the N-th dimension corresponding to samples.
"""
struct StackedArrayLayout <: DataLayout end

"""
    write_header!(io::IOStream, ::StackedArrayLayout, dims)

Utility function for writing the header of a binary file that stores data in
`StackedArrayLayout`. `dims` is a tuple of integers representing the dimensions
of the data.
"""
function write_header!(io::IOStream, ::StackedArrayLayout, dims)
    length(dims) >= 3 ||
        throw(ArgumentError("`StackedArrayLayout` must be given by at least three dimensional arrays"))
    write(io, length(dims) - 2)
    return write(io, [dims...])
end

"""
    read_header(io::IOStream, ::StackedArrayLayout)

Utility function for reading the header of a binary file that stores data in
`StackedArrayLayout`.
"""
function read_header(io::IOStream, ::StackedArrayLayout)
    nstate_dims = read(io, Int)
    state_dim = Int[]
    for _ in (1:nstate_dims)
        push!(state_dim, read(io, Int))
    end
    num_timepoints = read(io, Int)
    num_samples = read(io, Int)
    return Tuple(state_dim), num_timepoints, num_samples
end

"""
    state_dim(::Union{TimeseriesLayout,SingleObsTimeseriesLayout,
                      SingleTimeseriesLayout},
              data::AbstractArray)
              
    state_dim(layout::TimeseriesLayout, data::Vector{<:AbstractArray})

Return the dimension of the state space of the time series data.

!!! note
    The state space may in principle be represented by any `AbstractArray`s
    (i.e. not necessarily a `Vector`). Following the convention for `Base.size`
    the dimension is given as a tuple of integers.
"""
function state_dim(::Union{TimeseriesLayout,SingleObsTimeseriesLayout,
                           SingleTimeseriesLayout},
                   data::AbstractArray)
    return size(data)[1:(ndims(data) - 1)]
end
function state_dim(l::TimeseriesLayout, data::Vector{<:AbstractArray})
    return state_dim(l, first(data))
end
state_dim(::StackedArrayLayout, data::AbstractArray) = size(data)[1:(ndims(data) - 2)]

"""
    num_samples(layout::DataLayout, data::AbstractArray)

    num_samples(layout::TimeseriesLayout, data::Vector{<:AbstractArray})

    num_samples(layout::SingleTimeseriesLayout, data::AbstractArray)

Return the total number of samples in `data`.
"""
num_samples(::DataLayout, data::AbstractArray) = size(data, ndims(data))
function num_samples(::TimeseriesLayout, data::Vector{<:AbstractArray})
    return num_samples(StackedArrayLayout(), first(data))
end
num_samples(::SingleTimeseriesLayout, data::AbstractArray) = 1

"""
    num_timepoints(layout::TimeseriesLayout, data::Vector{<:AbstractArray})
    
    num_timepoints(layout::StackedArrayLayout, data::AbstractArray)

    num_timepoints(layout::SingleTimeseriesLayout, data::AbstractArray)

Return the number of time points in the time series `data`.
"""
num_timepoints(::TimeseriesLayout, data::Vector{<:AbstractArray}) = length(data)
num_timepoints(::StackedArrayLayout, data::AbstractArray) = size(data, ndims(data) - 1)
num_timepoints(::SingleTimeseriesLayout, data::AbstractArray) = size(data, ndims(data))

"""
    Base.reshape(data::AbstractArray, from_to::Pair{LayoutA, LayoutB})
                
Transform `data` stored in `LayoutA()` into a format following `LayoutB()`.
"""
function Base.reshape(data::AbstractArray,
                      ::Pair{StackedArrayLayout,TimeseriesLayout})
    timedim = ndims(data) - 1
    return copy.(eachslice(data; dims=timedim))
end

function _single_array_container_dims(dummy_data::AbstractArray, num_timepoints::Int)
    n = ndims(dummy_data)
    state_dim = size(dummy_data)[1:(n - 1)]
    num_samples = size(dummy_data, n)
    return (state_dim..., num_timepoints, num_samples)
end

function _single_array_container(dummy_data::AbstractArray, num_timepoints::Int)
    dims = _single_array_container_dims(dummy_data, num_timepoints)
    return similar(dummy_data, dims...)
end

function Base.reshape(data::Vector{<:AbstractArray},
                      ::Pair{TimeseriesLayout,StackedArrayLayout})
    out = _single_array_container(first(data), length(data))
    for (i, d) in enumerate(data)
        selectdim(out, n, i) .= d
    end
    return out
end

"""
    dimsofreshape(data::Vector{<:BatchDelimitedArray},
                  from_to::Pair{TimeseriesLayout,StackedArrayLayout})
                
Compute the size of the `Array` that would be needed to reshape `data` stored in
`TimeseriesLayout()` into a format following `StackedArrayLayout()`.

!!! tip
    Using this function, as opposed to calling `Base.reshape` might be handy if
    it's not possible to perform the reshaping in memory, and instead, an `Mmap`
    array needs to be constructed.
"""
function dimsofreshape(data::Vector{<:BatchDelimitedArray},
                       ::Pair{TimeseriesLayout,StackedArrayLayout})
    return _single_array_container_dims(getdata(first(data)), length(data))
end

function Base.reshape(data::Vector{<:BatchDelimitedArray},
                      ::Pair{TimeseriesLayout,StackedArrayLayout})
    out = _single_array_container(getdata(first(data)), length(data))
    for (i, d) in enumerate(data)
        selectdim(out, n, i) .= d
    end
    return out
end

function Base.reshape(data::AbstractArray,
                      ::Pair{SingleObsTimeseriesLayout,TimeseriesLayout})
    n = ndims(data)
    return reshape(data,
                   size(data)[1:(n - 1)],
                   1, # add time dimension
                   size(data, n))
end