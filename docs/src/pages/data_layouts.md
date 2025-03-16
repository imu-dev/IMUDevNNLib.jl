# Data Layouts

`DataLayout`s define conventions for storing data.

!!! note
    In `imu.dev` we are mostly interested in IMU data and these almost
    exclisively take the form of timeseries; however, in principle, any type of data can be represented with a suitable subtype of `DataLayout`.

```@docs
DataLayout
```

## Implemented layouts

```@docs
SingleTimeseriesLayout
TimeseriesLayout
SingleObsTimeseriesLayout
StackedArrayLayout
```

## Utility functions
As `DataLayout`s impose some conventions about how the data are organized, knowing the type of `DataLayout` is sometimes already enough to extract useful information without needing to know any further specifics. We implement a couple of such utility functions:

```@docs
state_dim
num_samples
num_timepoints(::TimeseriesLayout, data::Vector{<:AbstractArray})
```

Additionally, there are some further utility functions that make sense only under specific `DataLayout`s:

```@docs
write_header!
read_header
```

Finally, we provide routines to convert between different layouts:

```@docs
Base.reshape
dimsofreshape
```