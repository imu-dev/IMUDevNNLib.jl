# IMUDevNNLib.jl

<!-- [![][docs-stable-img]][docs-stable-url] -->
<!-- &nbsp;&nbsp;&nbsp;&nbsp; -->

[![][docs-dev-img]][docs-dev-url]
[![Build Status](https://github.com/imu-dev/IMUDevNNLib.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/imu-dev/IMUDevNNLib.jl/actions/workflows/CI.yml?query=branch%3Amain)

Toolbox designed to support workflows involving Neural Networks. [NNlib](https://github.com/FluxML/NNlib.jl) is the prototypical package of this kind, which covers most of the common use cases; however, when restricting to IMU-type data it is beneficial to encapsulate other, more specific sorts of abstractions. This package collects some of those.

> [!NOTE]
> See also [IMUDevNNTrainingLib](https://github.com/imu-dev/IMUDevNNTrainingLib.jl) for a dependant package that introduces further extensions (at the expense of heavier dependencies) that are useful at the time of training Neural Nets on IMU data.

> [!IMPORTANT]
> This package **<u>is not</u>** registered with Julia's [General Registry](https://github.com/JuliaRegistries/General), but instead, with `imu.dev`'s local [IMUDevRegistry](https://github.com/imu-dev/IMUDevRegistry). In order to use this package you will need to add [IMUDevRegistry](https://github.com/imu-dev/IMUDevRegistry) to the list of your registries.

<!-- [docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://imu-dev.github.io/IMUDevNNLib.jl/stable/ -->

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://imu-dev.github.io/IMUDevNNLib.jl/dev
