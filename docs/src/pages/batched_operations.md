# Batched Operations

[NNlib](https://github.com/FluxML/NNlib.jl) implements a useful spectrum of [operations designed to work with batches](https://fluxml.ai/NNlib.jl/dev/reference/#Batched-Operations). At the moment, it is mainly a syntactic sugar when deployed on a CPU (as it doesn't seem to be hooked up to [MKL](https://github.com/JuliaLinearAlgebra/MKL.jl)); however, when running code on a GPU, it seamlessly integrates with CUDA, making use of the [cuBLAS](https://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemmbatched)'s native batched operations.

Though the most common use cases are covered by [NNlib](https://github.com/FluxML/NNlib.jl) many aren't and it doesn't take long to encounter them. For now, this package implements some of the common batched operations encountered when working with IMU data that don't feature in [NNlib](https://github.com/FluxML/NNlib.jl); though only to the extent of being a syntactic sugar for operations on CPUs.

!!! danger "TODO"
    Once the `imu.dev` project reaches certain level of maturity we will aim to implement the most relevant batched operations on GPUs (by hooking to [cuBLAS](https://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemmbatched), which won't be difficult as [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) already does the heavy lifting), but for the time being, no concrete plans are made as to the exact date when that would happen.

## Matrix-vector multiplication

```@docs
batched_matvecmul
```

## Matrix inverse

```@docs
batched_rinv
batched_rinvsolve
batched_rinvsolve!
```
