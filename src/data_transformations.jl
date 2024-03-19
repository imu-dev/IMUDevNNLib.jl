"""
Parent to all preprocessing data transformations

# Callable functions

```julia
(transf::DataTransformation)([rng::Random.AbstractRNG], data::AbstractArray...)
```

Apply transformation `transf` to the `data`.
"""
abstract type DataTransformation end

function (transf::DataTransformation)(layout::DataLayout, data::AbstractArray...)
    return transf(Random.default_rng(), layout, data...)
end

"""
    IdentityTransformation <: DataTransformation

Identity transformation. Does nothing to the data that's passed through it.

# Callable functions

```julia
(::IdentityTransformation)([::Random.AbstractRNG],
                           ::DataTransformation,
                           data::AbstractArray...)
```

Pass `data` through the transformation. Does nothing.
"""
struct IdentityTransformation <: DataTransformation end

function (::IdentityTransformation)(::Random.AbstractRNG,
                                    ::DataTransformation,
                                    data::AbstractArray)
    return data
end

function (::IdentityTransformation)(::Random.AbstractRNG,
                                    ::DataTransformation,
                                    data::AbstractArray...)
    return [data...]
end

"""
    RandomPlanarRotation{T} <: DataTransformation

Random rotation of the data in the plane spanned by `indices`. The angle is
chosen uniformly from the interval `[-max_angle, max_angle]`.

# Callable functions

```julia
(transf::RandomPlanarRotation)([rng::Random.AbstractRNG],
                               ::SingleObsTimeseriesLayout,
                               data::AbstractMatrix...)
```

Each matrix `data` is rotated by the same angle.

```julia
(transf::RandomPlanarRotation)(rng::Random.AbstractRNG,
                               ::SingleArrayLayout,
                               data::Abstract3Tensor...)
```

Each sample in `data` is rotated by a different angle, note however, that each
tensor `data` is rotated by the same sequence of angles. In other words,
sample `i` from each tensor `data` is rotated by the same angle and sample
`j` from each tensor `data` is rotated by the same angle, but angles for
samples `i` and `j` (with `i≠j`) are different from one another.
"""
struct RandomPlanarRotation{T} <: DataTransformation
    max_angle::Float64
    indices::T

    function RandomPlanarRotation(max_angle::Real, indices::T=1:2) where {T}
        length(indices) == 2 ||
            throw(ArgumentError("The selected plane must be spanned by two indices"))
        return new{typeof(indices)}(max_angle, indices)
    end
end

# Transformation of a single obervation, i.e. each `data` has dimensions:
# state_space_dim × num_timepoints
function (transf::RandomPlanarRotation)(rng::Random.AbstractRNG,
                                        ::SingleObsTimeseriesLayout,
                                        data::AbstractMatrix...)
    angle = rand(rng) * transf.max_angle
    return map(d -> plane_rotation(d, angle; plane_indices=transf.indices), data)
end

function _draw_random_angle_per_sample(rng::Random.AbstractRNG,
                                       dummy_data::Abstract3Tensor,
                                       max_angle)
    # batch dimension should always be last irrespective of the layout
    batchsize = size(dummy_data, ndims(dummy_data))
    angles = similar(dummy_data, batchsize)
    rand!(rng, angles)
    angles *= eltype(dummy_data)(max_angle)
    return angles
end

# Transformation of multiple observations, i.e. each `data` has dimensions:
# state_space_dim × num_timpoints × batchsize
function (transf::RandomPlanarRotation)(rng::Random.AbstractRNG,
                                        ::SingleArrayLayout,
                                        data::Abstract3Tensor...)
    angles = _draw_random_angle_per_sample(rng, first(data), transf.max_angle)
    return map(d -> plane_rotation(d, angles; plane_indices=transf.indices), data)
end

"""
    RandomShift <: DataTransformation

Shift the data in the time dimension by a random amount chosen uniformly from
the interval `(-shift, shift)`. It will assume that the data passed through it
is a time series with the time dimension encompassing segment:

    `left_padding` | `window` | `right_padding`

where `left_padding` and `right_padding` are the same and equal to `shift`.

# Callable functions

```julia
(transf::RandomShift)(rng::Random.AbstractRNG,
                      ::SingleArrayLayout,
                      data::AbstractArray...)
```
"""
struct RandomShift <: DataTransformation
    shift::Int
end

function Base.rand(rng::Random.AbstractRNG, transf::RandomShift)
    return rand(rng, (-transf.shift):(transf.shift))
end

function fetch_frame(layout::SingleArrayLayout, data::AbstractArray, shift, max_shift)
    ntimepts = num_timepoints(layout, data)
    window = ntimepts - 2 * max_shift
    frame_id = max_shift + 1 + shift
    timedim = ndims(data) - 1
    return copy(selectdim(data, timedim, frame_id:(frame_id + window - 1)))
end

function (transf::RandomShift)(rng::Random.AbstractRNG,
                               layout::SingleArrayLayout,
                               data::AbstractArray)
    return fetch_frame(layout, data, rand(rng, transf), transf.shift)
end

function (transf::RandomShift)(rng::Random.AbstractRNG,
                               layout::SingleArrayLayout,
                               data::AbstractArray...)
    shift = rand(rng, transf)
    return map(d -> fetch_frame(layout, d, shift, transf.shift), data)
end
