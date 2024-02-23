module IMUDevNNLib

using NNlib

const Abstract3Tensor = AbstractArray{<:Any,3}

include("utils.jl")
include("batched_operations.jl")

# utils.jl
export skipfirstobs, eachslicelastdim

# batched_operations.jl
export batched_matvecmul, batched_rinv, batched_rinvsolve, batched_rinvsolve!

end
