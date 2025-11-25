using Revise
using Pkg

Pkg.activate("local")
luma_path = "/home/emmanuel/Packages/Luma.jl/"
Pkg.develop(path=luma_path)

#Pkg.instantiate()
Pkg.resolve()
Base.retry_load_extensions()
using Luma
using KernelAbstractions, Test, CUDA, BenchmarkTools
using AcceleratedKernels

n = 1000000
op = +
T = Float32
src_cpu = [rand() for i in (1:n)]
src = CuArray{T}(src_cpu)
dst = CuArray{T}([0 for _ in (1:n)])


CUDA.@sync Luma.scan!(op, dst, src)

isapprox(dst, CuArray{T}(accumulate(op, src_cpu)))
@test isapprox(dst, CuArray{T}(accumulate(op, src_cpu)))