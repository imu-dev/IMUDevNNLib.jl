"""
Parent to all `struct`s that slice and dice data for further analysis.
"""
abstract type DataArranger end

"""
    SlidingWindow

A data arranger that's a simple sliding window. It will cut segments of data
from the input array by sliding a `window` of a fixed size, moving it by
`stride` timepoints at a time. The `window`s can be additionally padded with
`pad` elements on both sides; note however, that this will be a squishy padding
on the edges of the data, meaning that only the `window` will be restricted to
the bounds of the original data, whereas the padding will be free to jump over
to the opposite site of the `window` (in such a way that `sum(pad)` is
unchanged) if it encounters an edge.

# Constructors

```julia
slidingwindow(T::DataType=Float32;
              stride::Int=10,
              window::Int=200,
              pad=(left=0, right=0))
```

A recommended way to construct a `SlidingWindow` data arranger. The `T` will be
used to specify the `eltype` of the arranged data.

# Function calls

```julia
(arranger::SlidingWindow)(io::IOStream,
                          from_to::Pair{SingleTimeseriesLayout,
                                        SingleArrayLayout},
                          data::AbstractArray)
```

Arrange the `data` using the `SlidingWindow` and write it to the `io` stream.

!!! tip
    This is intended for writing the data to a binary file for subsequent
    reading using `Mmap`.

```julia
(arranger::SlidingWindow)(from_to::Pair{SingleTimeseriesLayout,
                                        SingleArrayLayout},
                          data::AbstractArray)
```

Arrange the `data` using the `SlidingWindow` `arranger` and return it as a new
array.
"""
struct SlidingWindow{T} <: DataArranger
    stride::Int
    window::Int
    pad::NamedTuple{(:left, :right),Tuple{Int,Int}}
end

Base.eltype(::SlidingWindow{T}) where {T} = T

function slidingwindow(T::DataType=Float32;
                       stride::Int=10,
                       window::Int=200,
                       pad=(left=0, right=0))
    stride > 0 || throw(ArgumentError("`stride` must be > 0"))
    window > 0 || throw(ArgumentError("`window` must be > 0"))
    # I think it might make sense to actually allow to have them negative...
    # all(pad .>= 0) || throw(ArgumentError("`pad` must be non-negative"))
    if isa(pad, Number)
        isa(pad, Integer) || throw(ArgumentError("`pad` must be specified with integers"))
        pad = (left=pad, right=pad)
    else
        length(pad) == 2 ||
            throw(ArgumentError("`pad` must be either a number or a 2-length (named)tuple/vector"))
        pad = (left=pad[1], right=pad[2])
    end
    return SlidingWindow{T}(stride, window, pad)
end

"""
    padded_window(arranger::SlidingWindow)

Return the size of the window padded with `pad` elements on both sides.
"""
padded_window(arranger::SlidingWindow) = arranger.window + sum(arranger.pad)

"""
    padded_frame_indexrange(arranger::SlidingWindow, id::Int, max_len::Int)

Return the range of indices that correspond to the `id`-th frame of the
`SlidingWindow` `arranger`. These can be used to index into the original data
to extract the entire frame.
"""
function padded_frame_indexrange(arranger::SlidingWindow, id::Int, max_len::Int)
    len = padded_window(arranger)
    from_id = max(1, id - arranger.pad.left)
    to_id = min(max_len, from_id + len - 1)
    # if `to_id` bumps into `max_len`, then we need to correct `from_id`
    from_id = to_id - len + 1
    return from_id:to_id
end

"""
    frame_ids(arranger::SlidingWindow, max_len::Int)

List all the frame ids that can be extracted from the data of length `max_len`.
"""
function frame_ids(arranger::SlidingWindow, max_len::Int)
    pw = padded_window(arranger)
    max_len >= pw ||
        throw("Padded window of the `SlidingWindow` ($pw) exceeds the total number of timepoints in the dataset ($max_len)")
    return 1:(arranger.stride):(max_len - arranger.window + 1)
end

function (arranger::SlidingWindow)(io::IOStream,
                                   from_to::Pair{SingleTimeseriesLayout,
                                                 SingleArrayLayout},
                                   data::AbstractArray)
    from_layout, _ = from_to
    ntimepts = num_timepoints(layout, data)
    fids = frame_ids(arranger, ntimepts)
    dims = (state_dim(from_layout, data)..., padded_window(arranger), length(fids))
    write_header!(io, from_layout, dims)
    for (_, frame_id) in enumerate(fids)
        indices = padded_frame_indexrange(arranger, frame_id, ntimepts)
        write(io, eltype(arranger).(selectalonglastdim(data, indices)))
    end
    return nothing
end

function (arranger::SlidingWindow)(from_to::Pair{SingleTimeseriesLayout,
                                                 SingleArrayLayout},
                                   data::AbstractArray)
    from_layout, _ = from_to
    ntimepts = num_timepoints(from_layout, data)
    fids = frame_ids(arranger, ntimepts)
    d = state_dim(from_layout, data)
    out = Array{eltype(arranger),length(d) + 2}(undef,
                                                d...,
                                                padded_window(arranger),
                                                length(fids))

    for (i, frame_id) in enumerate(fids)
        indices = padded_frame_indexrange(arranger, frame_id, ntimepts)
        selectalonglastdim(out, i) .= selectalonglastdim(data, indices)
    end
    return out
end