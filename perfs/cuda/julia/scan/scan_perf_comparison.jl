using Revise
using Pkg

Pkg.activate("$(@__DIR__)/../../")

using Luma
using KernelAbstractions, CUDA, BenchmarkTools
using AcceleratedKernels
using Quaternions

#%% Basic Float32 scan
n = 1_000_000
T = Float32
FlagType = UInt8
op = +
src = CuArray{T}(1:n)
dst = CUDA.zeros(T, n)

start_time = time()
while time() - start_time < 0.5
    CUDA.@sync Luma.scan!(op, dst, src)
end
CUDA.@profile Luma.scan!(op, dst, src)

#buf = IOBuffer()
#CUDA.@device_code_ptx io = buf Luma.scan!(op, dst, src)
#asm = String(take!(copy(buf)))
#@test occursin("st.global.v4", asm)

#%% Without tmp allocation, UInt64 flags (no initialization needed)
n = 1_000_000
T = Float32
FlagType = UInt64
op = +
src = CuArray{T}(1:n)
dst = CUDA.zeros(T, n)

tmp = Luma.get_allocation(Luma.scan!, dst, src; FlagType=FlagType)
start_time = time()
while time() - start_time < 0.500
    CUDA.@sync Luma.scan!(op, dst, src; tmp=tmp, FlagType=FlagType)
end
CUDA.@profile Luma.scan!(op, dst, src; tmp=tmp, FlagType=FlagType)

#%% CUDA baseline
src = CuArray{T}(1:n)
dst = CUDA.zeros(T, n)

start_time = time()
while time() - start_time < 0.500
    CUDA.@sync CUDA.accumulate!(op, dst, src)
end
CUDA.@profile CUDA.accumulate!(op, dst, src)

#%% AcceleratedKernels baseline
start_time = time()
while time() - start_time < 0.500
    CUDA.@sync AcceleratedKernels.accumulate!(op, dst, src; init=zero(T))
end
CUDA.@profile AcceleratedKernels.accumulate!(op, dst, src; init=zero(T))

#%% ============= Float32, large n ============================
n = 100_000_000
T = Float32
op = +
src = CUDA.ones(T, n)
dst = CUDA.zeros(T, n)

start_time = time()
while time() - start_time < 0.500
    CUDA.@sync Luma.scan!(op, dst, src)
end
CUDA.@profile Luma.scan!(op, dst, src)

#%% Float64, large n
n = 100_000_000
T = Float64
op = +
src = CuArray{T}(1:n)
dst = CUDA.zeros(T, n)

start_time = time()
while time() - start_time < 0.500
    CUDA.@sync Luma.scan!(op, dst, src)
end
CUDA.@profile Luma.scan!(op, dst, src)

#%% CUDA baseline
start_time = time()
while time() - start_time < 0.500
    CUDA.@sync CUDA.accumulate!(op, dst, src)
end
CUDA.@profile CUDA.accumulate!(op, dst, src)

#%% AcceleratedKernels (default algorithm, not DecoupledLookback)
start_time = time()
while time() - start_time < 0.500
    CUDA.@sync AcceleratedKernels.accumulate!(op, dst, src; init=zero(T))
end
CUDA.@profile AcceleratedKernels.accumulate!(op, dst, src; init=zero(T))
#%% ====================     Quaternions ======================
n = 100_000
op = *
T = QuaternionF64
src_cpu = [QuaternionF64((x ./ sqrt(sum(x .^ 2)))...) for x in eachcol(randn(4, n))]
src = CuArray(src_cpu)
dst = CuArray(zeros(T, n))

start_time = time()
while time() - start_time < 0.500
    CUDA.@sync Luma.scan!(op, dst, src; Nitem=4)
end
CUDA.@profile Luma.scan!(op, dst, src; Nitem=4)

#%% CUDA baseline (requires commutativity - may fail for quaternions)
start_time = time()
while time() - start_time < 0.500
    CUDA.@sync CUDA.accumulate!(op, dst, src)
end
CUDA.@profile CUDA.accumulate!(op, dst, src)

#%% AcceleratedKernels (default algorithm, not DecoupledLookback)
start_time = time()
while time() - start_time < 0.500
    CUDA.@sync AcceleratedKernels.accumulate!(op, dst, src; init=zero(T))
end
CUDA.@profile AcceleratedKernels.accumulate!(op, dst, src; init=zero(T))

#%% ============================ UInt8 ================================
n = 100_000_000
op = +
T = UInt8
src = CUDA.ones(T, n)
dst = CUDA.zeros(T, n)

start_time = time()
while time() - start_time < 0.500
    CUDA.@sync Luma.scan!(op, dst, src)
end
CUDA.@profile Luma.scan!(op, dst, src)

#%% CUDA baseline
start_time = time()
while time() - start_time < 0.500
    CUDA.@sync CUDA.accumulate!(op, dst, src)
end
CUDA.@profile CUDA.accumulate!(op, dst, src)


#%% AcceleratedKernels (default algorithm, not DecoupledLookback)
start_time = time()
while time() - start_time < 0.500
    CUDA.@sync AcceleratedKernels.accumulate!(op, dst, src; init=zero(T))
end
CUDA.@profile AcceleratedKernels.accumulate!(op, dst, src; init=zero(T))