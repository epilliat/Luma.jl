import Base: copy!

function vcopy!(dst::AbstractGPUVector, src::AbstractGPUVector; Nitem=4)
    backend = get_backend(src)
    ndrange = fld(length(src), Nitem)
    copy_kernel!(backend)(dst, src, Val(Nitem); ndrange=ndrange)
end

function setvalue!(dst::AbstractGPUVector{T}, val::T; Nitem=4) where T
    backend = get_backend(dst)
    ndrange = fld(length(dst), Nitem)
    setvalue_kernel!(backend)(dst, val, Val(Nitem); ndrange=ndrange)
end