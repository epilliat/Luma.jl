function mapreduce1d! end

#const DEFAULT_MAPREDUCE_NITEM = Dict(
#    UInt8 => 16
#)
@inline function default_nitem(::typeof(mapreduce1d!), ::Type{T}) where {T}
    if sizeof(T) == 1
        return 8
    elseif sizeof(T) == 2
        return 4
    else
        return 1
    end
end

function get_allocation(
    ::typeof(mapreduce1d!),
    srcs::NTuple{U,AbstractGPUArray{T}};
    #
    blocks::Integer,
    H::Type{HT},
    FlagType::Type{FT}
) where {U,T,HT,FT}
    backend = get_backend(srcs[1])
    sz = sum(get_partition_sizes(blocks, HT, FT))
    return KernelAbstractions.allocate(backend, UInt8, sz)
end

function mapreduce1d!(
    f::F, op::O,
    dst::AbstractGPUArray{S},
    srcs::NTuple{U,AbstractGPUArray{T}};
    #
    g::G=identity,
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    config=nothing,
    FlagType::Type{FT}=UInt8
) where {U,S,T,F<:Function,O<:Function,G<:Function,FT}
    n = length(srcs[1])
    backend = get_backend(srcs[1])
    H = Base.promote_op(f, T)

    if isnothing(Nitem)
        Nitem = default_nitem(mapreduce1d!, T)
    end

    if isnothing(config)
        #kernel = mapreduce1d_kernel!(backend, 10000, 100000) # dummy high values for launch config
        #dummy_flag_array = KernelAbstractions.allocate(backend, FlagType, 0)
        #dummy_partial = dst # we could put KernelAbstractions.allocate(backend, H, 0) for more accuracy
        config = (workgroup=256, blocks=50)#get_default_config(kernel, f, op, dst, srcs, g, Val(Nitem), dummy_partial, dummy_flag_array, FlagType(0)) #time costly with @eval only the first time, then cached
    end

    workgroup, blocks = config
    workgroup = min(workgroup, n)
    ndrange = min(blocks * workgroup, max((fld(n, workgroup)) * workgroup, 1))
    # ensure that ndrange * Nitem <= N. Take a smaller Nitem if necessary (take power of two for alignment safety)
    Nitem = min(Nitem, prevpow(2, max(fld(n, ndrange), 1)))

    if isnothing(tmp)
        tmp = get_allocation(mapreduce1d!, srcs; blocks=blocks, H=H, FlagType=FT)
    end

    partial, flag = partition(tmp, blocks, H, FT)

    if FT == UInt8
        setvalue!(flag, 0x00; Nitem=8)
        targetflag = 0x01
    else
        targetflag = rand(FT)
    end

    mapreduce1d_kernel!(backend, workgroup, ndrange)(f, op, dst, srcs, g, Val(Nitem), partial, flag, targetflag)
end

function mapreduce1d!(
    f, op,
    dst::AbstractGPUArray{S},
    src::AbstractGPUArray{T};
    #
    g=identity,
    Nitem=nothing,
    tmp::Union{AbstractGPUArray{UInt8},Nothing}=nothing,
    config=nothing,
    FlagType::Type{FT}=UInt8
) where {S,T,FT}
    return mapreduce1d!(f, op, dst, (src,); g=g, tmp=tmp, config=config, FlagType=FT, Nitem=Nitem)
end