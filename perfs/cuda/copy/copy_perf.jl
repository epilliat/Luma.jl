using Revise
using Pkg
Pkg.activate("/home/emmanuel/Packages/Luma.jl/dev")
#Pkg.develop(path="/home/emmanuel/Packages/MemoryAccess.jl/")
#Pkg.develop(path="/home/emmanuel/Packages/Luma.jl")
#Pkg.instantiacte()
Base.retry_load_extensions()
using MemoryAccess
using Luma
using KernelAbstractions, Test, CUDA, BenchmarkTools

using Quaternions




#%%
n = 100000000
T = Float32
src_cpu = [1.0f0 for i in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])
Nitem = 1

start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    Luma.copy!(dst, src, Nitem=Nitem)
end

prof = CUDA.@profile Luma.copy!(dst, src; Nitem=Nitem)

#%%
n = 1000000
T = Float32
src_cpu = [1.0f0 for i in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])
Nitem = 4

start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    Luma.copy!(dst, src, Nitem=Nitem)
end

prof = CUDA.@profile Luma.copy!(dst, src; Nitem=Nitem)


#%%


n = 1000000
T = UInt8
src_cpu = [1.0f0 for i in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])
Nitem = 8 #better

start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    Luma.copy!(dst, src, Nitem=Nitem)
end

prof = CUDA.@profile Luma.copy!(dst, src; Nitem=Nitem)

#%%
n = 1000000
T = UInt8
src_cpu = [1.0f0 for i in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])
Nitem = 4

start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    Luma.copy!(dst, src, Nitem=Nitem)
end

prof = CUDA.@profile Luma.copy!(dst, src; Nitem=Nitem)







#================== setvalue! =========================#
#useful for rapid flag initialization !
using Luma
n = 1000000
T = UInt8
dst = CuArray{T}([1 for _ in (1:n)])
Nitem = 8 #better

start_time = time()
while time() - start_time < 0.500  # 500ms warm-up
    Luma.setvalue!(dst, T(1), Nitem=Nitem)
end

prof = CUDA.@profile Luma.setvalue!(dst, T(1); Nitem=Nitem)

