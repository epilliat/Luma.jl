using Revise
using Pkg

Pkg.activate("$(@__DIR__())/../")
luma_path = "$(@__DIR__())/../../../"
#Pkg.develop(path=luma_path)

#Pkg.instantiate()
using Luma
using KernelAbstractions, Test, CUDA, BenchmarkTools
using AcceleratedKernels
using Quaternions



n = 1000000
T = Float32
FlagType = UInt8
op = +
f = identity
src = CuArray{T}([i for i in (1:n)])
dst = CuArray{T}([0])

start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync Luma.mapreduce!(f, op, dst, src)
end
CUDA.@profile Luma.mapreduce!(f, op, dst, src)

#%% not counting tmp memory allocation, and not initializing flags (default behavior for UInt64 flags)
n = 1000000
T = Float32
FlagType = UInt64
op = +
Nitem = 8
f = identity
src = CuArray{T}([i for i in (1:n)])
dst = CuArray{T}([0 for _ in (1:n)])

start_time = time()
tmp = get_allocation(Luma.mapreduce1d!, f, op, dst, (src,); FlagType=FlagType, Nitem=Nitem)

while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync Luma.mapreduce!(f, op, dst, src; tmp=tmp, FlagType=FlagType, Nitem=4)
end
CUDA.@profile Luma.mapreduce!(f, op, dst, src; tmp=tmp, FlagType=FlagType, Nitem=4)





#%% CUDA
start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync mapreduce(f, op, src)
end
prof = CUDA.@profile mapreduce(f, op, src)


#%% Accelerated Kernels
start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync AcceleratedKernels.mapreduce(f, op, src; init=T(0))
end
prof = CUDA.@profile AcceleratedKernels.mapreduce(f, op, src; init=T(0))


#%%============= Float64, larger n ============================
n = 1000000
T = Float64
FlagType = UInt8
op = +
f = identity
src = CuArray{T}([i for i in (1:n)])
dst = CuArray{T}([0])

start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync Luma.mapreduce!(f, op, dst, src, Nitem=4)
end
CUDA.@profile Luma.mapreduce!(f, op, dst, src, Nitem=4)



#%% CUDA
start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync mapreduce(f, op, src)
end
prof = CUDA.@profile mapreduce(f, op, src)


#%% Accelerated Kernels
start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync AcceleratedKernels.mapreduce(f, op, src; init=T(0))
end
prof = CUDA.@profile AcceleratedKernels.mapreduce(f, op, src; init=T(0))




#%%============================ UInt8 ================================

n = 100000000
op = +
f = identity
T = UInt8
src_cpu = [0x01 for _ in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])

start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    CUDA.@sync Luma.mapreduce!(f, op, dst, src, Nitem=16)
end
CUDA.@profile Luma.mapreduce!(f, op, dst, src, Nitem=16)
#%%

start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync mapreduce(f, op, src)
end
prof = CUDA.@profile mapreduce(f, op, src)

#%%
#%% Accelerated Kernels
start_time = time()
while time() - start_time < 0.5  # 500ms warm-up
    CUDA.@sync AcceleratedKernels.mapreduce(f, op, src; init=T(0))
end
prof = CUDA.@profile AcceleratedKernels.mapreduce(f, op, src; init=T(0))

