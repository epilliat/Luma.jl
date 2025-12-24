@kernel inbounds = true function mapreduce2d_ver_splitwarp_kernel!(#vertical reduction, square matrix (or horizontal-rectangular)
    f::F, op::O,
    dst::AbstractArray{S},
    srcs::NTuple{U,AbstractArray{T}},
    g::G,
    ::Val{Nitem},
    ::Val{Nthreads},
    H::DataType #No partial here to get intermediary type
) where {F<:Function,O<:Function,G<:Function,U,T,S,Nitem,Nthreads}
    n, p = size(srcs[1])
    workgroup = Int(@groupsize()[1])
    ndrange = Int(@ndrange()[1])


    blocks = cld(ndrange, workgroup)

    lid = Int(@index(Local, Linear))
    gid = Int(@index(Group, Linear))
    wid = cld(lid, warpsz) # warp id

    I = (gid - 1) * workgroup + lid
    chid = cld(I, Nthreads) #id of the chunk

    lane = (lid - 1) % warpsz + 1
    chlane = (lid - 1) % Nthreads + 1


    id_base = fld((chid - 1) * n, Nitem) + chlane
    if id_base * Nitem <= chid * n
        values = broadcast_apply_across(f, srcs, id_base, Val(Nitem))
        if chlane == 1
            j0 = (chid - 1) * n - fld((chid - 1) * n, Nitem) * Nitem + 1
            val = values[j0]
            for j in j0+1:Nitem
                val = op(val, values[j])
            end
        else
            val = tree_reduce(op, values)
        end
    else
        val = f(srcs[1][1]) #dummy value for optimization purpose
    end
    i = id_base + Nthreads
    while i * Nitem <= chid * n
        values = broadcast_apply_across(f, srcs, i, Val(Nitem))
        val = op(val, tree_reduce(op, values))
        i += Nthreads
    end
    id_base = (i - 1) * Nitem + 1
    if id_base <= chid * n
        for j in id_base:chid*n
            values = broadcast_apply_across(f, srcs, j, Val(1))
            val = op(val, tree_reduce(op, values))
        end
    end

    offset = 1
    while offset < Nthreads
        shuffled = @shfl(Up, val, offset, warpsz, 0xffffffff)
        if chlane > offset
            val = op(shuffled, val)
        end
        offset <<= 1
    end
    #val = op(val, f.(vload(srcs[1], i, Val(Nitem)))...)
    if chlane == Nthreads && chlane <= n || chlane == n
        dst[chid] = g(val)
    end
end