
# ============================================================================
# Configuration
# ============================================================================

T = Float32
f(x) = x
op = +

# ============================================================================
# Helpers
# ============================================================================

function make_test_arrays(n::Int, p::Int; seed::Int=42)
    Random.seed!(seed)
    src = CuArray{T}(rand(T, n, p))
    dst = CuArray{T}(zeros(T, 1, p))
    return src, dst
end

function check_result(dst, src)
    expected = mapreduce(f, op, src; dims=1)
    return isapprox(Array(dst), Array(expected))
end

# ============================================================================
# Test suite
# ============================================================================

@testset "mapreduce2d_ver! default config" begin

    # ------------------------------------------------------------------
    # Square-ish matrices
    # ------------------------------------------------------------------
    @testset "square matrices" begin
        for (n, p) in [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]
            src, dst = make_test_arrays(n, p)
            Luma.mapreduce2d_ver!(f, op, dst, (src,))
            @test check_result(dst, src)
        end
    end

    # ------------------------------------------------------------------
    # Tall matrices (n >> p): many rows, few columns
    # ------------------------------------------------------------------
    @testset "tall matrices (n >> p)" begin
        for (n, p) in [(1024, 10), (4096, 32), (8192, 64), (16384, 100), (65536, 16)]
            src, dst = make_test_arrays(n, p)
            Luma.mapreduce2d_ver!(f, op, dst, (src,))
            @test check_result(dst, src)
        end
    end

    # ------------------------------------------------------------------
    # Wide matrices (p >> n): few rows, many columns
    # ------------------------------------------------------------------
    @testset "wide matrices (p >> n)" begin
        for (n, p) in [(3, 10_000), (16, 10_000), (32, 50_000), (64, 100_000), (128, 100_000), (256, 100_000)]
            src, dst = make_test_arrays(n, p)
            Luma.mapreduce2d_ver!(f, op, dst, (src,))
            @test check_result(dst, src)
        end
    end

    # ------------------------------------------------------------------
    # Small n (triggers horizontal rectangular heuristics)
    # ------------------------------------------------------------------
    @testset "small n" begin
        for (n, p) in [(4, 1000), (8, 1000), (16, 1000), (32, 1000), (48, 1000), (63, 1000)]
            src, dst = make_test_arrays(n, p)
            Luma.mapreduce2d_ver!(f, op, dst, (src,))
            @test check_result(dst, src)
        end
    end

    # ------------------------------------------------------------------
    # Large n (may trigger multi-block per column)
    # ------------------------------------------------------------------
    @testset "large n" begin
        for (n, p) in [(2048, 1000), (4096, 500), (8192, 200), (16384, 100), (32768, 50)]
            src, dst = make_test_arrays(n, p)
            Luma.mapreduce2d_ver!(f, op, dst, (src,))
            @test check_result(dst, src)
        end
    end

    # ------------------------------------------------------------------
    # Edge cases: single row/column, minimal sizes
    # ------------------------------------------------------------------
    @testset "edge cases" begin
        # Single column
        src, dst = make_test_arrays(256, 1)
        Luma.mapreduce2d_ver!(f, op, dst, (src,))
        @test check_result(dst, src)

        # Single row (degenerate but should work)
        src, dst = make_test_arrays(1, 1000)
        Luma.mapreduce2d_ver!(f, op, dst, (src,))
        @test check_result(dst, src)

        # Minimal
        src, dst = make_test_arrays(1, 1)
        Luma.mapreduce2d_ver!(f, op, dst, (src,))
        @test check_result(dst, src)

        # Power of 2 boundaries
        for n in [31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
            src, dst = make_test_arrays(n, 100)
            Luma.mapreduce2d_ver!(f, op, dst, (src,))
            @test check_result(dst, src)
        end
    end

    # ------------------------------------------------------------------
    # Non-power-of-2 dimensions
    # ------------------------------------------------------------------
    @testset "non-power-of-2" begin
        for (n, p) in [(100, 100), (300, 500), (777, 333), (1000, 1000), (1234, 5678)]
            src, dst = make_test_arrays(n, p)
            Luma.mapreduce2d_ver!(f, op, dst, (src,))
            @test check_result(dst, src)
        end
    end

    # ------------------------------------------------------------------
    # Stress test: large total elements
    # ------------------------------------------------------------------
    @testset "stress test" begin
        # ~100M elements
        src, dst = make_test_arrays(1024, 100_000)
        Luma.mapreduce2d_ver!(f, op, dst, (src,))
        @test check_result(dst, src)

        # ~50M elements, tall
        src, dst = make_test_arrays(50_000, 1000)
        Luma.mapreduce2d_ver!(f, op, dst, (src,))
        @test check_result(dst, src)
    end

end