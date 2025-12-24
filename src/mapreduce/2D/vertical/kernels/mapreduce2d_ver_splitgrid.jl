@kernel unsafe_indices = false function mapreduce2d_ver_splitgrid_kernel!(#vertical reduction, square matrix (or horizontal-rectangular)
    f::F, op::O,
    dst::AbstractArray{S},
    srcs::NTuple{U,AbstractArray{T}},
    g::G,
    ::Val{Nitem},
    ::Val{Nblocks}, #Nthreads = workgroup* Nblocks
    partial::AbstractArray{H},
    flag::AbstractArray{FlagType},
    targetflag::FlagType
) where {F<:Function,O<:Function,G<:Function,U,T,H,S,Nitem,Nblocks,FlagType}
    n, p = size(srcs[1])
    workgroup = Int(@groupsize()[1])
    Nthreads = workgroup * Nblocks
    ndrange = @ndrange()[1]

    blocks = cld(ndrange, workgroup)

    lid = Int(@index(Local, Linear))
    gid = Int(@index(Group, Linear))
    wid = cld(lid, warpsz) # warp id

    I = (gid - 1) * workgroup + lid
    chid = cld(I, Nthreads) #id of the chunk

    lane = (lid - 1) % warpsz + 1

    shared = @localmem H 32

    local_gid = (gid - 1) % Nblocks + 1 # block index relative to the column
    id_base = fld((chid - 1) * n, Nitem) + (local_gid - 1) * workgroup + lid

    if id_base * Nitem <= chid * n
        values = broadcast_apply_across(f, srcs, id_base, Val(Nitem))
        if lane == 1
            j0 = (chid - 1) * n - fld((chid - 1) * n, Nitem) * Nitem + 1
            val = values[j0]
            for j in j0+1:Nitem
                val = op(val, values[j])
            end
        else
            val = tree_reduce(op, values)
        end
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

    @warpreduce(val, lane, op)

    if lane == warpsz # Nitem * workgroup * Nblocks <= n
        shared[wid] = val
    end
    @synchronize

    if wid == 1
        val_acc = shared[lane]
        @warpreduce(val_acc, lane, op)
        if lane == min(cld(workgroup, warpsz), cld(n, warpsz))
            if Nblocks == 1
                dst[chid] = val_acc
            else
                idx = (chid - 1) * Nblocks + local_gid
                partial[idx] = val_acc
                @access flag[idx] = targetflag
            end
        end
    end
    if Nblocks > 1 && local_gid == 1
        i = lid
        idx = (chid - 1) * Nblocks + i
        while i <= Nblocks
            (@access flag[idx]) == targetflag && break
        end
        i <= Nblocks && (val = partial[idx])
        i += workgroup
        while i <= Nblocks
            idx = (chid - 1) * Nblocks + i
            while true
                (@access flag[idx]) == targetflag && break
            end
            val = op(val, partial[idx])
            i += workgroup
        end
        @warpreduce(val, lane, op)
        if lane == warpsz && lid <= Nblocks || lid == Nblocks
            shared[wid] = val
        end
        @synchronize
        if wid == 1#cld(workgroup, warpsz) && wid <= cld(blocks, warpsz) || wid == cld(blocks, warpsz)
            val_acc = shared[lane]
            @warpreduce(val_acc, lane, op)
            if lane == min(cld(workgroup, warpsz), cld(Nblocks, warpsz))
                dst[chid] = g(val_acc)
            end
        end
    end
end