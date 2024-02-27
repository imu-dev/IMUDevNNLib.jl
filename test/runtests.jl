using IMUDevNNLib
using LinearAlgebra
using Test

approx_equal(a, b; atol=1e-7) = maximum(norm.(a .- b)) < atol

@testset "IMUDevNNLib.jl" begin
    @testset "utils.jl" begin
        a = rand(3, 4, 5)
        @test skipfirstobs(a) == a[:, :, 2:end]
        @test eachslicelastdim(a) == eachslice(a; dims=3)

        b = rand(1:7...)
        @test skipfirstobs(b) == b[:, :, :, :, :, :, 2:end]
        @test eachslicelastdim(b) == eachslice(b; dims=7)
    end
    @testset "temporal_data.jl" begin
        a = rand(3, 4, 5)
        b = rand(3, 4, 5)
        td = temporal_data(Float32, a, b; skipfirst=true)
        td2 = temporal_data(Float32.(a[:, :, 2:end]), Float32.(b[:, :, 2:end]))
        @test td.xx == td2.xx
        @test td.yy == td2.yy
        @test IMUDevNNLib.MLUtils.numobs(td) == 4
        # get time series for the first batch
        x₀, y₀, xx, yy = IMUDevNNLib.MLUtils.getobs(td, 1)
        @test approx_equal(x₀, a[:, 1, 1])
        @test approx_equal(y₀, b[:, 1, 1])
        @test approx_equal(IMUDevNNLib.MLUtils.getobs(td, 1)[3],
                           [x for x in eachslice(a[:, 1, 2:end]; dims=2)])
        @test approx_equal(IMUDevNNLib.MLUtils.getobs(td, 1)[4],
                           [x for x in eachslice(b[:, 1, 2:end]; dims=2)])
    end
end
