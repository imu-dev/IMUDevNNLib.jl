"""
    plane_rotation(data::AbstractMatrix, angle; plane_indices=1:2)

Rotate the `data` in the plane spanned by indices `plane_indices` by `angle`.
The data are assumed to be given in a matrix form, with:
1. first dimension indexing through the state coordinates, and
2. the second dimension indexing through the time points

```julia
plane_rotation(data::Abstract3Tensor, angles::AbstractVector; plane_indices=1:2)
```

Rotate the `data` in the plane spanned by indices `plane_indices` by `angles`.
The data are assumed to be given in a 3-tensor form, with:
1. first dimension indexing through the state coordinates,
2. the second dimension indexing through the samples, and
3. the third dimension indexing through the time points

"""
function plane_rotation(data::AbstractMatrix, angle; plane_indices=1:2)
    rot = [cos(angle) -sin(angle);
           sin(angle) cos(angle)]
    rot_data = copy(data)
    rot_data[plane_indices, :] .= rot * rot_data[plane_indices, :]

    return rot_data
end

function plane_rotation(data::Abstract3Tensor, angles::AbstractVector; plane_indices=1:2)
    length(plane_indices) == 2 ||
        throw(ArgumentError("The selected plane must be spanned by two indices"))
    i1, i2 = plane_indices
    out = copy(data)
    ca = cos.(angles)
    sa = sin.(angles)

    d1 = @view data[i1, :, :]
    d2 = @view data[i2, :, :]
    out[i1, :, :] = d1 .* ca' .- d2 .* sa'
    out[i2, :, :] = d1 .* sa' .+ d2 .* ca'
    return out
end