using Test, CUDA, Luma

@testset "Luma.scan! basic tests" begin
    # Basic multiplication test
    n = 100_000
    T = Float32
    src_cpu = rand(T, n)
    src = CuArray(src_cpu)
    dst = CUDA.zeros(T, n)

    CUDA.@sync Luma.scan!(*, dst, src)
    @test isapprox(dst, CuArray(accumulate(*, src_cpu)))

    # Small array test
    n = 1005
    T = Float64
    src_cpu = rand(T, n)
    src = CuArray(src_cpu)
    dst = CUDA.zeros(T, n)

    CUDA.@sync Luma.scan!(identity, *, dst, src)
    @test isapprox(dst, CuArray(accumulate(*, src_cpu)))
end

@testset "Luma.scan! with Quaternions (non-commutative)" begin
    using Quaternions

    n = 1_000_000
    T = QuaternionF64
    # Normalized quaternions
    src_cpu = [QuaternionF64((x ./ sqrt(sum(x .^ 2)))...) for x in eachcol(randn(4, n))]
    src = CuArray(src_cpu)
    dst = CuArray(zeros(T, n))  # zeros(T, n) works for Quaternions

    CUDA.@sync Luma.scan!(identity, *, dst, src)
    @test isapprox(dst, CuArray(accumulate(*, src_cpu)))
end

@testset "Luma.scan! comprehensive tests" begin
    test_sizes = [1, 10, 100, 1_000, 10_000, 1_000_000]
    test_types = [Float64, Int32]
    test_ops = [
        (+, "addition"),
        (max, "maximum"),
        (min, "minimum"),
    ]

    for T in test_types
        for n in test_sizes
            for (op, op_name) in test_ops
                @testset "T=$T, n=$n, op=$op_name" begin
                    for trial in 1:5
                        if T <: AbstractFloat
                            src_cpu = rand(T, n)
                        else
                            src_cpu = T[rand(1:10) for _ in 1:n]
                        end

                        src = CuArray(src_cpu)
                        dst = CUDA.zeros(T, n)

                        CUDA.@sync Luma.scan!(op, dst, src)
                        expected = accumulate(op, src_cpu)

                        if T <: AbstractFloat
                            @test isapprox(Array(dst), expected)
                        else
                            @test Array(dst) == expected
                        end
                    end
                end
            end
        end
    end
end

@testset "UInt8 scan tests" begin
    test_sizes = [10_001]
    test_ops = [(+, "addition"), (max, "maximum"), (min, "minimum")]

    for n in test_sizes
        for (op, op_name) in test_ops
            @testset "n=$n, op=$op_name" begin
                for trial in 1:10
                    src_cpu = rand(UInt8, n)
                    src = CuArray(src_cpu)
                    dst = CUDA.zeros(UInt8, n)

                    CUDA.@sync Luma.scan!(op, dst, src)
                    @test Array(dst) == accumulate(op, src_cpu)
                end
            end
        end
    end
end