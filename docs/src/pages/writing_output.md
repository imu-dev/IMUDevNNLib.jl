# Writing output

Once the training and testing is done and we have a model we wish to use in production we often need to save the results.
The prediction happening in production stage may still happen in batches, but we often wish to save the data in a common [`StackedArrayLayout`](@ref).
To this end, we can make use of a utility struct `BatchDelimitedArray` to save to and read from using iteration that iterates over batches:

```@docs
BatchDelimitedArray
```

Rather than accessing data through:

```julia
a::BatchDelimitedArray
a.data
```

it is recommended to use `getdata` function instead:

```@docs
getdata
```

