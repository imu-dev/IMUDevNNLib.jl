"""
    TimeseriesIterator

A joint iterator over two iterables `xx` and `yy` that conceptually correspond
to "underlying state" and "observations". Use `timeseriesiterator` to construct
this `struct`.

# Constructors

```julia
timeseriesiterator(xx::Vector{<:AbstractArray},
                   yy::Vector{<:AbstractArray},
                   policy;
                   num_timepoints::Int=num_timepoints(policy,
                                                      TimeseriesLayout(),
                                                      xx,
                                                      yy))
```

A recommended constructor for `TimeseriesIterator`. `num_timepoints` is the
total number of timepoints in a timeseries. `policy` defines the way in which
joint iteration over `xx` and `yy` should be carried out.

!!! note
    Usually `num_timepoints(TimeseriesLayout(), xx) == num_timepoints(TimeseriesLayout(), yy) == num_timepoints`,
    but in general this need not be the case if a custom iteration `policy` is defined.
"""
struct TimeseriesIterator{FT<:Iterators.Stateful,LT<:Iterators.Stateful,P}
    num_timepoints::Int
    xx::FT
    yy::LT
    policy::P
end

function timeseriesiterator(xx::Vector{<:AbstractArray},
                            yy::Vector{<:AbstractArray},
                            policy=ZipIterPolicy();
                            num_timepoints::Int=num_timepoints(policy,
                                                               TimeseriesLayout(),
                                                               xx,
                                                               yy))
    return TimeseriesIterator(num_timepoints,
                              Iterators.Stateful(xx),
                              Iterators.Stateful(yy),
                              policy)
end

function Base.iterate(iter::TimeseriesIterator, state=1)
    if state > iter.num_timepoints
        return nothing
    end
    xx = takexx(iter.policy, state) ? popfirst!(iter.xx) : nothing
    yy = takeyy(iter.policy, state) ? popfirst!(iter.yy) : nothing
    return (xx, yy), state + 1
end

"""
    Iterators.reset!(i::TimeseriesIterator)

Reset the internal iterators to a starting position.
"""
function Iterators.reset!(i::TimeseriesIterator)
    Iterators.reset!(i.xx)
    Iterators.reset!(i.yy)
    return nothing
end