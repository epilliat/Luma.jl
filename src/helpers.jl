# Helper function for kernels when working with Tuples of vectors

@generated function apply(f, srcs::NTuple{U}, idx) where {U}
    args = [:(srcs[$u][idx]) for u in 1:U]
    return :(f($(args...)))
end









function get_partition_sizes(blocks, Types::Type...)
    return (((blocks * sizeof(T) + 8 * sizeof(T)) >> 3) << 3 for T in Types)
end

function partition(tmp::AbstractVector{UInt8}, blocks, Types...)
    sizes = get_partition_sizes(blocks, Types...)
    accum_sizes = (0, accumulate(+, sizes)...)

    return (
        reinterpret(T, view(tmp, accum_sizes[i]+1:accum_sizes[i+1]))
        for (i, T) in enumerate(Types)
    )
end


function get_default_config(obj::KernelAbstractions.Kernel, args...)
    return (workgroup=256, blocks=100)
end
# These are just default values, but we should write specific methods in function of each backend to maximize occupency when writting persistent kernels -- see CUDA extension



function get_default_config_cached(obj::K, args...) where {K<:KernelAbstractions.Kernel}
    param_types = [:(::$(typeof(arg))) for arg in args]
    config = get_default_config(obj, args...)

    @eval get_default_config_cached(::$K, $(param_types...)) = $config

    return config
end


@generated function default_value(::Type{T}) where T
    if isbitstype(T)
        nbytes = Base.packedsize(T)
        if nbytes == 0
            return :(error("Cannot create instance of zero-sized type $T"))
        end
        bytes = ntuple(i -> zero(UInt8), Val(nbytes))
        # Create from zero bits
        u = reinterpret(T, bytes)
        return quote
            $u
        end
    else
        return :(error("Cannot create default value for non-bitstype $T"))
    end
end