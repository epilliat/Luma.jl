function mapreduce1d! end


function get_allocation(
    ::typeof(mapreduce1d!),
    f, op,
    dst::AbstractGPUArray{Outf},
    srcs::NTuple{U,AbstractGPUArray{T}};
    #
    Nitem=1,
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    config::Union{NamedTuple,Nothing}=nothing,
    FlagType=UInt8
) where {U,Outf,T}
    backend = get_backend(dst)
    if isnothing(config)
        kernel = mapreduce1d_kernel!(backend, 10000, 100000) # dumy high values for launch config
        dummy_flag_array = KernelAbstractions.allocate(backend, FlagType, 0)
        config = get_default_config_cached(kernel, f, op, dst, srcs, Val(Nitem), dst, dummy_flag_array, FlagType(0))
    end
    sz = sum(get_partition_sizes(config.blocks, Outf, FlagType))
    return KernelAbstractions.allocate(backend, UInt8, sz)
end

function mapreduce1d!(
    f, op,
    dst::AbstractGPUArray{Outf},
    srcs::NTuple{U,AbstractGPUArray{T}};
    #
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    Nitem::Integer=1,
    config=nothing,
    FlagType=UInt8
) where {U,Outf,T}

    n = length(srcs[1])
    backend = get_backend(srcs[1])

    if isnothing(config)
        kernel = mapreduce1d_kernel!(backend, 10000, 100000) # dumy high values for launch config
        dummy_flag_array = KernelAbstractions.allocate(backend, FlagType, 0)
        config = get_default_config_cached(kernel, f, op, dst, srcs, Val(Nitem), dst, dummy_flag_array, FlagType(0)) #time costly with @eval only the first time, then cached
        #println((MemoryAccess.InteractiveUtils.@which get_default_config_cached(kernel, f, op, dst, srcs, dst, dummy_flag_array, FlagType(0))))
    end

    workgroup, blocks = config

    workgroup = min(workgroup, n)
    ndrange = min(blocks * workgroup, max((fld(n, workgroup)) * workgroup, 1))

    # ensure that ndrange * Nitem <= N. Take a smaller Nitem if necessary (take power of two for alignment safety)
    Nitem = min(Nitem, prevpow(2, max(fld(n, ndrange), 1)))


    if isnothing(tmp)
        tmp = get_allocation(mapreduce1d!, f, op, dst, srcs; FlagType=FlagType, config=config, Nitem=Nitem)
    end
    partial, flag = partition(tmp, blocks, Outf, FlagType)
    if FlagType == UInt8
        setvalue!(flag, 0x00; Nitem=8)
        targetflag = 0x01
    else
        targetflag = rand(FlagType)
    end
    KernelAbstractions.synchronize(backend)
    mapreduce1d_kernel!(backend, workgroup, ndrange)(f, op, dst, srcs, Val(Nitem), partial, flag, targetflag)

end

function mapreduce1d!(
    f, op,
    dst::AbstractGPUArray{Outf},
    src::AbstractGPUArray{T};
    #
    Nitem=1,
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    config=nothing,
    FlagType=UInt8
) where {Outf,T}
    return mapreduce1d!(f, op, dst, (src,); tmp=tmp, config=config, FlagType=FlagType, Nitem=Nitem)
end
