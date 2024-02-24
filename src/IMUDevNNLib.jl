module IMUDevNNLib

using NNlib
using MLUtils

const Abstract3Tensor = AbstractArray{<:Any,3}

include("utils.jl")
include("batched_operations.jl")
include("temporal_data.jl")

# temporal_data.jl
export TemporalData, temporal_data, num_samples, num_timepoints, state_dim, obs_dim

# utils.jl
export skipfirstobs, eachslicelastdim, selectfirstobs

# batched_operations.jl
export batched_matvecmul, batched_rinv, batched_rinvsolve, batched_rinvsolve!

end
