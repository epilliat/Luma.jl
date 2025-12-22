function scan! end
# ============================================================================
# Configuration helpers
# ============================================================================

@inline function default_nitem(::typeof(scan!), ::Type{T}) where {T}
    sz = sizeof(T)
    if sz == 1
        return 32
    elseif sz == 2
        return 16
    elseif sz == 4
        return 8
    elseif sz == 8 #Float64
        return 8
    else
        return 4
    end
end

const DEFAULT_SCAN_CONFIG = (workgroup=256,)

@inline function get_scan_config(n::Int, Nitem::Int, workgroup::Int)
    ndrange = cld(n, Nitem)
    blocks = cld(ndrange, workgroup)
    return ndrange, blocks
end

# ============================================================================
# Allocation API
# ============================================================================

function get_allocation(
    ::typeof(scan!),
    dst::AbstractGPUVector{Outf},
    src::AbstractGPUVector{T};
    Nitem::Union{Integer,Nothing}=nothing,
    config::Union{NamedTuple{(:workgroup,)},Nothing}=nothing,
    FlagType::Type{FT}=UInt8
) where {Outf,T,FT}
    _Nitem = Nitem === nothing ? default_nitem(scan!, Outf) : Int(Nitem)
    workgroup = config === nothing ? DEFAULT_SCAN_CONFIG.workgroup : config.workgroup

    n = length(src)
    _, blocks = get_scan_config(n, _Nitem, workgroup)

    backend = get_backend(dst)
    sz = sum(get_partition_sizes(blocks, Outf, Outf, FT))
    return KernelAbstractions.allocate(backend, UInt8, sz)
end

# ============================================================================
# Public API
# ============================================================================

function scan!(
    op,
    dst::AbstractGPUVector{Outf},
    src::AbstractGPUVector{T};
    tmp::Union{AbstractGPUVector{UInt8},Nothing}=nothing,
    Nitem::Union{Integer,Nothing}=nothing,
    config::Union{NamedTuple{(:workgroup,)},Nothing}=nothing,
    FlagType::Type{FT}=UInt8
) where {Outf,T,FT}
    return scan!(identity, op, dst, src; tmp, Nitem, config, FlagType)
end

function scan!(
    f::F, op::O,
    dst::AbstractGPUVector{Outf},
    src::AbstractGPUVector{T};
    tmp::Union{AbstractGPUVector{UInt8},Nothing}=nothing,
    Nitem::Union{Integer,Nothing}=nothing,
    config::Union{NamedTuple{(:workgroup,)},Nothing}=nothing,
    FlagType::Type{FT}=UInt8
) where {Outf,T,F<:Function,O<:Function,FT}
    n = length(src)
    n == 0 && return dst

    backend = get_backend(src)

    # Resolve defaults (avoiding `something` for type stability)
    _Nitem = Nitem === nothing ? default_nitem(scan!, Outf) : Int(Nitem)
    workgroup = config === nothing ? DEFAULT_SCAN_CONFIG.workgroup : config.workgroup

    # Compute launch configuration
    ndrange, blocks = get_scan_config(n, _Nitem, workgroup)

    # Allocate temporaries if not provided
    _tmp = if tmp === nothing
        sz = sum(get_partition_sizes(blocks, Outf, Outf, FT))
        KernelAbstractions.allocate(backend, UInt8, sz)
    else
        tmp
    end

    _scan_impl!(f, op, dst, src, Val(_Nitem), _tmp, ndrange, blocks, workgroup, Outf, FT, n, backend)
end

# ============================================================================
# Core implementation
# ============================================================================

function _scan_impl!(
    f::F, op::O,
    dst::AbstractGPUVector{Outf},
    src::AbstractGPUVector{T},
    ::Val{Nitem},
    tmp::AbstractGPUVector{UInt8},
    ndrange::Int,
    blocks::Int,
    workgroup::Int,
    ::Type{H},
    ::Type{FT},
    n::Int,
    backend
) where {Outf,T,F,O,Nitem,H,FT}
    partial1, partial2, flag = partition(tmp, blocks, H, H, FT)

    # Initialize flags and select target value (type-stable)
    targetflag = if FT === UInt8
        setvalue!(flag, FT(0); Nitem=8)
        FT(1)
    else
        rand(FT)
    end

    scan_kernel!(backend, workgroup)(
        f, op, dst, src, Val(Nitem), partial1, partial2, flag, targetflag;
        ndrange=ndrange
    )

    return dst
end