using Test
using CUDA
using Luma
import InteractiveUtils: @which

@testset "CUDA" begin
    @testset "copy" begin
        include("cuda/copy_test.jl")
    end
    @testset "scan" begin
        include("cuda/scan_test.jl")
    end
    @testset "mapreduce" begin
        include("cuda/mapreduce_test.jl")
    end
end
