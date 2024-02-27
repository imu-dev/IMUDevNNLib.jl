# Temporal Data

Passing a time-series data through a Neural Network often requires a choice of an architecture that falls under the category of the so-called [Recurrent Neural Networks (RNNs)](https://www.ibm.com/topics/recurrent-neural-networks). [Flux](https://github.com/FluxML/Flux.jl/tree/master) package (that the project `imu.dev` predominantly relies upon for the task of training Neural Nets) requires these data to adhere to a [certain format](https://fluxml.ai/Flux.jl/stable/models/recurrence/). For this purpose we define a `TemporalData` struct to provide a unified way of handling training/validation/testing data to be passed through RNNs.
```@docs
TemporalData
```
## Data Loader
The `TemporalData` implements the interface required by [DataLoader](https://fluxml.ai/Flux.jl/previews/PR1786/data/dataloader/), which means it can be passed to it directly
```@docs
IMUDevNNLib.MLUtils.numobs
IMUDevNNLib.MLUtils.getobs
```
### Example
```julia
data = ...
td = temporal_data(Float32,
                   data.train.xx,
                   data.train.yy; skipfirst=true)

loader = Flux.DataLoader(td; batchsize=20, shuffle=true)
```
## Utility functions
Utility functions include
```@docs
num_samples
num_timepoints
state_dim
obs_dim
Base.size(td::TemporalData, of_what::Symbol; kwargs...)
```

# BatchedTemporalContainer

Even though `TemporalData` is the preferred format for working with RNNs, in the end, after we've run our forecasting algorithm we often want to come back to the usual format of one big N+2 dimensional matrix (`N > 0`), where `N` is the dimension of the state (usually `1` as we work with vectors) and where the last two dimensions are the batch (or samples) and time dimensions respectively. For this reason we define a `BatchedTemporalContainer` that facilitates easy writing of such data already at thetime of running our forecasting/testing algorithms.

```@docs
BatchedTemporalContainer
```

The container can be allocated directly from the `TemporalData` object by calling:

```@docs
outputcontainer
```

And the data can be retrieved with:

```@docs
getdata
```

### Example
Here is an example of how to use it:
```julia
model = ...
function smooth(loader::Flux.DataLoade{<:TemporalData})
    out = out_container(loader.data, :xx; batchsize=loader.batchsize)

    for (i, (x₀, _, _, yy)) in enumerate(loader)
        Flux.reset!(model)
        init!(model, x₀)
        # batch `i` timepoint `1`
        out[i, 1] .= x₀
        for (t, y) in enumerate(yy)
            x̂ₜ = model(y)
            # batch `i` timepoint `t+1`
            out[i, t + 1] = x̂ₜ
        end
    end

    return getdata(out)
end
```