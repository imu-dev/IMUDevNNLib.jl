using Documenter
using IMUDevNNLib

makedocs(; sitename="IMUDevNNLib",
         format=Documenter.HTML(),
         modules=[IMUDevNNLib],
         checkdocs=:exports,
         pages=["Home" => "index.md",
                "Manual" => ["Data Layouts" => joinpath("pages", "data_layouts.md"),
                             "Arrangers" => joinpath("pages", "data_arrangers.md"),
                             "Transformations" => joinpath("pages",
                                                           "data_transformations.md"),
                             "Iteration" => joinpath("pages", "timeseries_iteration.md"),
                             "Batched Operations" => joinpath("pages",
                                                              "batched_operations.md"),
                             "Other" => joinpath("pages", "other.md")]])

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(; repo="github.com/imu-dev/IMUDevNNLib.jl.git")
