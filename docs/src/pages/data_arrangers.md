# Data arrangers

Data arrangers are used to get from the point of having raw data to data that are split into something resembling samples.

```@docs
DataArranger
```

## Implemented arrangers

Currently only a single `DataArranger` is implemented: `SlidingWindow`. It's very commonly employed to chop down a long recording into many short chunks that can be subsequently passed to a Neural Network.

```@docs
SlidingWindow
```
