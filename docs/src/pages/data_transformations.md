# Data transformations

Data transformation are very commonly employed in Machine Learning as a way to augment the training data.

```@docs
DataTransformation
```

## Implemented data transformations

```@docs
IdentityTransformation
RandomPlanarRotation
RandomShift
```

`RandomPlanarRotation` will call function `plane_rotation` under the hood:

```@docs
plane_rotation
```