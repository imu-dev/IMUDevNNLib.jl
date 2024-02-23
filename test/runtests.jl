using IMUDevNNLib
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
end
