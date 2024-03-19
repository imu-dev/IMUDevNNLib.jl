using IMUDevNNLib
using LinearAlgebra
using Test

approx_equal(a, b; atol=1e-7) = maximum(norm.(a .- b)) < atol

@testset "IMUDevNNLib.jl" begin
    @testset "utils.jl" begin
        a = rand(3, 4, 5)
        @test skipfirstobs(a; timedim=3) == a[:, :, 2:end]
        b = rand(1:7...)
        @test skipfirstobs(b) == b[:, :, :, :, :, 2:end, :]
    end
end
