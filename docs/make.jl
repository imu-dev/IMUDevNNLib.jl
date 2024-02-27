using Documenter
using IMUDevNNLib

makedocs(; sitename="IMUDevNNLib",
         format=Documenter.HTML(),
         modules=[IMUDevNNLib],
         checkdocs=:exports,
         pages=["Home" => "index.md",
                "Manual" => ["Temporal Data" => joinpath("pages", "temporal_data.md"),
                             "Batched Operations" => joinpath("pages",
                                                              "batched_operations.md"),
                             "Other" => joinpath("pages", "other.md")]])

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
