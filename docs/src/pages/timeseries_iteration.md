# Iteration over timeseries data
In order to pass timeseries data through a recurrent neural networks we must iterate over timepoints and pass them one by one. Further complication may arise if the timeseries coming from `features` and that coming from `labels` should not be zipped together in a 1-to-1 manner. To this end we introduce an interface for `TimeseriesIterationPolicy`.

## Interface for iteration policy

```@docs
takexx
takeyy
num_timepoints
```

!!! note "Notation"
    `features` (or `input`) and `labels` come from the machine learning terminology, whereas `xx` and `yy` are used in State Space Models.
    The mapping between the two can be unintuitive. For instace, consider the following example from IMU data:

    !!! tip "Example"
        We observe `acceleration` and `angular velocity` to predict `position`.

    From the machine learning perspective we have:
    
    - `features`: the observations, i.e. `acceleration` and `angular velocity`
    - `labels`: the things we wish to predict i.e. `position` 
  
    On the other hand, from the perspective of state space modelling:

    - `xx`: the "underlying state", i.e. `position`
    - `yy`: the observations, i.e. `acceleration` and `angular velocity`
  
    So, technically, the mapping between two should be
    
    - `features` ⟷ `yy`
    - `labels` ⟷ `xx`
  
    However, `TimeseriesIterationPolicy` does not enforce this mapping and the user is free to pass whatever as `xx` or `yy`.
    They just need to make sure that the policy they apply to will act on the appropriate containers that they've passed.
    
    In fact, it is not uncommon to, rather confusingly use the notation `x` and `y` (or `X` and `Y`) also in machine learning and use a mapping:

    - `features` ⟷ `x`
    - `labels` ⟷ `y`
    
    We do not commit to any particular convention. In any example the adopted convention should follow easily from its context.

## Implemented policies

```@docs
ZipIterPolicy
TakeX0IterPolicy
TakeX0NoY0Policy
```

## Iteration

To iterate over two containers according to some `policy` use `TimeseriesIterator`:

```@docs
TimeseriesIterator
```

The following utility function is implemented for the `TimeseriesIterator`:

```@docs
Iterators.reset!
```

## Example
A very common workflow when working with Recurrent Neural Networks is demonstrated with the following pseudo-code:

```julia
data_loader = ...
preprocess!(...) = ...
model = ...
init_model! = ...
loss = ...
do_something_with_loss = ...

for (features, labels) in data_loader
    preprocess!(features, labels)

    features = reshape(features, SingleArrayLayout() => TimeseriesLayout())
    labels = reshape(input, SingleArrayLayout() => TimeseriesLayout())

    it = timeseriesiterator(features, labels, ZipIterPolicy())

    (x₀, y₀), it = Iterators.peel(it)
    init_model!(model, x₀)

    for (x, y) in it
        y° = model(x)
        l = loss(y, y°)
        do_something_with_loss(...)
    end
end
```