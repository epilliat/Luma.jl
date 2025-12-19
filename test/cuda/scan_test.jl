n = 100000
op(x...) = *(x...)
T = Float32
src_cpu = [rand() for i in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])

using Quaternions # Non commuative operation and larger structure
@test (@which Luma.scan!) == Luma
CUDA.@sync Luma.scan!(op, dst, src)
@test isapprox(dst, CuArray{T}(accumulate(op, src_cpu)))


#%%

n = 1005
op = *
T = Float64
src_cpu = [rand() for i in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])

CUDA.@sync Luma.scan!(identity, op, dst, src)
@test isapprox(dst, CuArray{T}(accumulate(op, src_cpu)))
#%%

n = 1000000
op = *
T = QuaternionF64
src_cpu = [QuaternionF64(x ./ sqrt(sum(x .^ 2))...) for x in eachcol(randn(4, n))]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])

CUDA.@sync Luma.scan!(identity, op, dst, src)
@test isapprox(dst, CuArray{T}(accumulate(op, src_cpu)))
#isapprox(accumulate(op, src), CuArray{T}(accumulate(op, src_cpu))) #fails: accumulate from CUDA currently requires commutativity contrary to Luma!

#%%



using Test, CUDA, Luma

@testset "Luma.scan! comprehensive tests" begin
    # Test different array sizes
    test_sizes = [
        1,
        10,           # Very small
        100,          # Small
        1_000,        # Medium
        10_000,       # Large
        1000_000
    ]

    # Test different types
    test_types = [Float64, Int32]

    # Test different operations
    test_ops = [
        (+, "addition"),
    ]

    for T in test_types
        for n in test_sizes
            for (op, op_name) in test_ops
                @testset "T=$T, n=$n, op=$op_name" begin
                    # Run 10 times to check stability
                    for trial in 1:5
                        # Generate random data appropriate for the type and operation
                        if T <: AbstractFloat
                            if op == *
                                # For multiplication, use smaller values to avoid overflow
                                src_cpu = T[rand() * 0.1 + 0.9 for i in 1:n]
                            else
                                src_cpu = T[rand() for i in 1:n]
                            end
                        else  # Integer types
                            if op == *
                                # For integer multiplication, use very small values
                                src_cpu = T[rand(1:2) for i in 1:n]
                            elseif op in (min, max)
                                src_cpu = T[rand(1:100) for i in 1:n]
                            else
                                src_cpu = T[rand(1:10) for i in 1:n]
                            end
                        end

                        src = CuArray{T}(src_cpu)
                        dst = CuArray{T}(zeros(T, n))

                        # Run the scan
                        CUDA.@sync Luma.scan!(op, dst, src)

                        # Compute expected result
                        expected = CuArray{T}(accumulate(op, src_cpu))

                        # Check correctness
                        if T <: AbstractFloat
                            @test isapprox(dst, expected)
                        else
                            @test dst == expected
                        end
                    end
                end
            end
        end
    end

    # Additional test with transform function
    @testset "scan! with transform function" begin
        for trial in 1:10
            n = 1_005
            T = Float64
            op = *

            src_cpu = [rand() * 0.1 + 0.9 for i in 1:n]
            src = CuArray{T}(src_cpu)
            dst = CuArray{T}(zeros(T, n))

            CUDA.@sync Luma.scan!(identity, op, dst, src)
            expected = CuArray{T}(accumulate(op, src_cpu))

            @test isapprox(dst, expected)
        end
    end

end

@testset "UInt8 scan tests" begin
    test_sizes = [10001]

    # Operations that work well with UInt8
    test_ops = [
        (+, "addition"),
        (max, "maximum"),
        (min, "minimum"),
    ]
    for n in test_sizes
        for (op, op_name) in test_ops
            @testset "n=$n, op=$op_name" begin
                for trial in 1:10
                    # Generate random UInt8 data
                    src_cpu = UInt8[rand(UInt8) for i in 1:n]
                    src = CuArray{UInt8}(src_cpu)
                    dst = CuArray{UInt8}(zeros(UInt8, n))

                    # Run the scan
                    CUDA.@sync Luma.scan!(op, dst, src)

                    # Compute expected result
                    expected = CuArray{UInt8}(accumulate(op, src_cpu))

                    # Check correctness (exact for UInt8)
                    @test dst == expected
                end
            end
        end
    end
end