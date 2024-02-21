using IMUDevNNLib
using LinearAlgebra
using Test

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
        @test td.x == td2.x
        @test td.y == td2.y
        @test IMUDevNNLib.MLUtils.numobs(td) == 4
        @test maximum(norm.(IMUDevNNLib.MLUtils.getobs(td, 1)[1] .-
                            [x for x in eachslice(a[:, 1, 2:end]; dims=2)])) < 1e-7
        @test maximum(norm.(IMUDevNNLib.MLUtils.getobs(td, 1)[2] .-
                            [x for x in eachslice(b[:, 1, 2:end]; dims=2)])) < 1e-7
    end
end
