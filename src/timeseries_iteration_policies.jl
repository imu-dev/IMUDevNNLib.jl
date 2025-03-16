"""
    takexx(policy, iter_state::Int)

Return a `boolean` for whether to take the next `xx` at iteration `iter_state`.
"""
takexx

"""
    takeyy(policy, iter_state::Int)

Return a `boolean` for whether to take the next `yy` at iteration `iter_state`.
"""
takeyy

"""

    num_timepoints(iter_policy, d::DataLayout, xx, yy)

Compute the number of timepoints for a given iteration policy `p`, data layout
`d` and containers `xx` and `yy`.
"""
num_timepoints

"""
    $(TYPEDEF)

A default iteration policy for [`TimeseriesIterator`](@ref) that zips all
`xx` and `yy` in a one-to-one fashion.
"""
struct ZipIterPolicy end
takexx(::ZipIterPolicy, ::Int) = true
takeyy(::ZipIterPolicy, ::Int) = true
function num_timepoints(p::ZipIterPolicy,
                        l::TimeseriesLayout,
                        xx::Vector{<:AbstractArray},
                        yy::Vector{<:AbstractArray})
    n_xx = num_timepoints(l, xx)
    n_yy = num_timepoints(l, yy)
    n_xx == n_yy ||
        throw(ArgumentError("Number of timepoints in `xx` and `yy` are expected to be the same for policy choice $p"))
    return n_xx
end

"""
    $(TYPEDEF)

An iteration policy for [`TimeseriesIterator`](@ref) that takes the first
underlying state `x₀`, but discards all the rest and takes all states in `yy`.
"""
struct TakeX0IterPolicy end
takexx(::TakeX0IterPolicy, iter_state::Int) = iter_state == 1
takeyy(::TakeX0IterPolicy, ::Int) = true
function num_timepoints(p::TakeX0IterPolicy,
                        l::TimeseriesLayout,
                        xx::Vector{<:AbstractArray},
                        yy::Vector{<:AbstractArray})
    n_xx = num_timepoints(l, xx)
    n_xx >= 1 ||
        throw(ArgumentError("`xx` must contain at least one point under policy choice $p"))
    n_yy = num_timepoints(l, yy)
    n_yy >= 1 ||
        throw(ArgumentError("`yy` must contain at least one point under policy choice $p"))
    return n_yy
end

"""
    $(TYPEDEF)

An iteration policy for [`TimeseriesIterator`](@ref) that takes the first
underlying state `x₀`, but discards all the rest, as well as assumes that `yy`
has no observation `y₀` that would correspond to `x₀`.
"""
struct TakeX0NoY0Policy end
takexx(::TakeX0NoY0Policy, iter_state::Int) = iter_state == 1
takeyy(::TakeX0NoY0Policy, iter_state::Int) = iter_state != 1
function num_timepoints(p::TakeX0NoY0Policy,
                        l::TimeseriesLayout,
                        xx::Vector{<:AbstractArray},
                        yy::Vector{<:AbstractArray})
    n_xx = num_timepoints(l, xx)
    n_xx >= 1 ||
        throw(ArgumentError("`xx` must contain at least one point under policy choice $p"))
    n_yy = num_timepoints(l, yy)
    return n_yy + 1
end
