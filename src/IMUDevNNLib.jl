module IMUDevNNLib

using MLUtils

const Abstract3Tensor = AbstractArray{<:Any,3}

include("utils.jl")
include("temporal_data.jl")
include("batched_operations.jl")

export Abstract3Tensor

# utils.jl
export skipfirstobs, eachslicelastdim

# temporal_data.jl
export TemporalData, temporal_data

# batched_operations.jl
export batched_matvecmul, batched_rinv, batched_rinvsolve, batched_rinvsolve!

end
