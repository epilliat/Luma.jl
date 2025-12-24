"""
    mapreduce2d_ver!(f, op, dst, src; kwargs...)
    mapreduce2d_ver!(f, op, dst, srcs::NTuple; kwargs...)

GPU parallel 2D vertical map-reduce operation (reduces along rows).

# Keyword Arguments
- `g=identity`: Optional post-reduction transformation
- `tmp=nothing`: Pre-allocated temporary buffer (from `get_allocation`)
- `Nitem=nothing`: Number of items per thread (auto-selected based on element size if nothing)
- `Nthreads=nothing`: Number of threads per column reduction
- `Nblocks=nothing`: Number of blocks per column (>1 triggers splitgrid with decoupled lookback)
- `config=nothing`: Launch configuration as `(workgroup=W, blocks=B)` NamedTuple
- `FlagType=UInt8`: Type for synchronization flags
"""
function mapreduce2d_ver! end

# ============================================================================
# Configuration helpers
# ============================================================================

@inline function default_nitem(::typeof(mapreduce2d_ver!), ::Type{T}) where {T}
    if sizeof(T) == 1
        return 16
    elseif sizeof(T) == 2
        return 8
    else
        return 1
    end
end

const DEFAULT_MAPREDUCE2D_CONFIG = (workgroup=256, blocks=100)

# Main public entry point for allocation
function get_allocation(
    fn::typeof(mapreduce2d_ver!),
    srcs::NTuple{U,AbstractArray{T}};
    Nblocks::Integer,
    eltype::Union{Type,Nothing}=nothing,
    FlagType::Type{FT}=UInt8
) where {U,T,FT}
    H = something(eltype, T)
    return _get_allocation(fn, srcs, Nblocks, H, FT)
end

# Core implementation (positional args)
function _get_allocation(
    ::typeof(mapreduce2d_ver!),
    srcs::NTuple{U,AbstractArray{T}},
    Nblocks::Integer,
    ::Type{H},
    ::Type{FT}
) where {U,T,H,FT}
    Nblocks > 1 || error("Nblocks must be > 1, otherwise tmp allocation is unnecessary")
    backend = get_backend(srcs[1])
    _, p = size(srcs[1])
    sz = sum(get_partition_sizes(Nblocks * p, H, FT))
    return KernelAbstractions.allocate(backend, UInt8, sz)
end

# ============================================================================
# Public API (kwargs wrappers)
# ============================================================================

# Single array convenience wrapper
function mapreduce2d_ver!(
    f, op,
    dst::AbstractArray{S},
    src::AbstractArray{T};
    g=identity,
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    Nthreads=nothing,
    Nblocks=nothing,
    config=nothing,
    FlagType::Type{FT}=UInt8
) where {S,T,FT}
    return mapreduce2d_ver!(f, op, dst, (src,); g, tmp, Nitem, Nthreads, Nblocks, config, FlagType)
end

# Main public entry point
function mapreduce2d_ver!(
    f::F, op::O,
    dst::AbstractArray{S},
    srcs::NTuple{U,AbstractArray{T}};
    g::G=identity,
    tmp::Union{AbstractArray{UInt8},Nothing}=nothing,
    Nitem=nothing,
    Nthreads=nothing,
    Nblocks=nothing,
    config=nothing,
    FlagType::Type{FT}=UInt8
) where {U,S,T,F<:Function,O<:Function,G<:Function,FT}
    n, p = size(srcs[1])
    backend = get_backend(srcs[1])
    H = Base.promote_op(f, T)

    # Resolve config defaults
    _config = something(config, DEFAULT_MAPREDUCE2D_CONFIG)

    _mapreduce2d_ver_impl!(f, op, g, dst, srcs, Nitem, Nthreads, Nblocks, _config, tmp, H, FT, n, p, backend)
end

# ============================================================================
# Core implementation (positional args for type stability)
# ============================================================================

function _mapreduce2d_ver_impl!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    srcs::NTuple{U,AbstractArray{T}},
    Nitem::Union{Int,Nothing},
    Nthreads::Union{Int,Nothing},
    Nblocks::Union{Int,Nothing},
    config::NamedTuple{(:workgroup, :blocks)},
    tmp::Union{AbstractArray{UInt8},Nothing},
    ::Type{H},
    ::Type{FT},
    n::Int,
    p::Int,
    backend
) where {U,S,T,F,O,G,H,FT}
    workgroup, blocks = config.workgroup, config.blocks

    # Adjust workgroup to fit problem size
    workgroup = min(workgroup, prevpow(2, n * p))
    def_nitem = default_nitem(mapreduce2d_ver!, T)

    # Resolve Nthreads and Nitem with fine-tuning heuristics
    _Nthreads, _Nitem = _resolve_thread_item_config(Nthreads, Nitem, def_nitem, n)

    # Resolve Nblocks
    _Nblocks = something(Nblocks, min(cld(blocks, p), max(fld(n, workgroup), 1)))

    # Validation
    @assert _Nthreads * _Nitem <= n "Nthreads * Nitem must be <= n"
    @assert _Nblocks * _Nthreads * _Nitem <= n "Nblocks * Nthreads * Nitem must be <= n"
    if _Nblocks > 1
        @assert _Nthreads == workgroup "Nthreads must equal workgroup when Nblocks > 1"
    end

    ndrange = _Nthreads * _Nblocks * p

    _dispatch_mapreduce2d_ver!(
        f, op, g, dst, srcs,
        _Nitem, _Nthreads, _Nblocks,
        workgroup, ndrange, tmp, H, FT, p, backend
    )
end

# ============================================================================
# Thread/item configuration resolution
# ============================================================================

function _resolve_thread_item_config(
    Nthreads::Union{Int,Nothing},
    Nitem::Union{Int,Nothing},
    def_nitem::Int,
    n::Int
)
    if isnothing(Nthreads) && isnothing(Nitem)
        # Auto-tune based on problem size
        thresh = prevpow(2, max(fld(n, 4), 1))
        if thresh >= 256
            _Nitem = def_nitem
            _Nthreads = 256
        else
            # Horizontal rectangular case: fewer threads, more items
            _Nitem = min(thresh, def_nitem * 4)
            _Nthreads = cld(thresh, _Nitem)
        end
        # Ensure alignment constraint
        _Nitem = min(_Nitem, prevpow(2, max(fld(n, _Nthreads), 1)))
        return _Nthreads, _Nitem
    else
        # Both must be provided
        @assert !isnothing(Nthreads) && !isnothing(Nitem) "Must provide both Nthreads and Nitem, or neither"
        return Nthreads, Nitem
    end
end

# ============================================================================
# Kernel dispatch
# ============================================================================

function _dispatch_mapreduce2d_ver!(
    f::F, op::O, g::G,
    dst::AbstractArray{S},
    srcs::NTuple{U,AbstractArray{T}},
    Nitem::Int,
    Nthreads::Int,
    Nblocks::Int,
    workgroup::Int,
    ndrange::Int,
    tmp::Union{AbstractArray{UInt8},Nothing},
    ::Type{H},
    ::Type{FT},
    p::Int,
    backend
) where {U,S,T,F,O,G,H,FT}
    if Nblocks == 1
        # Single block per column: use warp or block reduction
        if Nthreads <= warpsz
            mapreduce2d_ver_splitwarp_kernel!(backend, workgroup, ndrange)(
                f, op, dst, srcs, g, Val(Nitem), Val(Nthreads), H
            )
        else
            mapreduce2d_ver_splitblock_kernel!(backend, workgroup, ndrange)(
                f, op, dst, srcs, g, Val(Nitem), Val(Nthreads), H
            )
        end
    else
        # Multiple blocks per column: use decoupled lookback
        _tmp = something(tmp, _get_allocation(mapreduce2d_ver!, srcs, Nblocks, H, FT))
        partial, flag = partition(_tmp, Nblocks * p, H, FT)

        # Initialize flags and select target value
        if FT === UInt8
            setvalue!(flag, 0x00; Nitem=8)
            targetflag = 0x01
        else
            targetflag = rand(FT)
        end

        mapreduce2d_ver_splitgrid_kernel!(backend, workgroup, ndrange)(
            f, op, dst, srcs, g, Val(Nitem), Val(Nblocks), partial, flag, targetflag
        )
    end
end