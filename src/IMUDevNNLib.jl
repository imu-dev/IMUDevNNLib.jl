module IMUDevNNLib

# I removed NNlib from dependencies, because I can't remember if it's being used
# anywhere. If it is the code will break at some point and I will add it back.
# using NNlib
using MLUtils
using Random

const Abstract3Tensor = AbstractArray{<:Any,3}

include("utils.jl")
include("batched_operations.jl")
include("batch_delimited_array.jl")
include("data_layouts.jl")
include("data_transformations.jl")
include("transformations.jl")
include("timeseries_iteration_policies.jl")
include("timeseries_iterator.jl")
include("data_arrangers.jl")

# utils.jl
export skipfirstobs, selectfirstobs, peelfirstobs, selectalonglastdim,
       mergealonglastdim, nnflatten

# batched_operations.jl
export batched_matvecmul, batched_rinv, batched_rinvsolve, batched_rinvsolve!

# batch_delimited_array.jl
export BatchDelimitedArray, batchdelimitedarray, getdata

# data_layouts.jl
export DataLayout, TimeseriesLayout, SingleObsTimeseriesLayout,
       SingleArrayLayout, state_dim, num_samples, num_timepoints,
       SingleTimeseriesLayout, write_header!, read_header, dimsofreshape

# data_transformations.jl
export DataTransformation, IdentityTransformation, RandomPlanarRotation,
       RandomShift

# transformationsl.jl
export plane_rotation

# timeseries_iteration_policies.jl
export ZipIterPolicy, TakeX0IterPolicy, TakeX0NoY0Policy, takexx, takeyy,
       num_timepoints

# timeseries_iterator.jl
export TimeseriesIterator, timeseriesiterator

# data_arrangers.jl
export DataArranger, SlidingWindow, slidingwindow

end
