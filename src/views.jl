"""
    H_symmetric_views{T}

Views into the unreduced KKT matrix
"""
struct H_unreduced_views{T}
    xx::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    xy::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    yx::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    xLzL::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    zLxL::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    xUzU::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    zUxU::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    zLzL::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    zUzU::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    yalyal::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
end

function H_unreduced_views(H::SparseMatrixCSC,idx::Indices)
    xx = view(H,idx.x,idx.x)
    xy = view(H,idx.x,idx.y)
    yx = view(H,idx.y,idx.x)
    xLzL = view(H,CartesianIndex.(idx.xL,idx.zL))
    zLxL = view(H,CartesianIndex.(idx.zL,idx.xL))
    xUzU = view(H,CartesianIndex.(idx.xU,idx.zU))
    zUxU = view(H,CartesianIndex.(idx.zU,idx.xU))
    zLzL = view(H,CartesianIndex.(idx.zL,idx.zL))
    zUzU = view(H,CartesianIndex.(idx.zU,idx.zU))
    yalyal = view(H,CartesianIndex.(idx.y_al,idx.y_al))

    H_unreduced_views(xx,xy,yx,xLzL,zLxL,xUzU,zUxU,zLzL,zUzU,yalyal)
end

"""
    H_symmetric_views{T}

Views into the symmetric KKT matrix
"""
struct H_symmetric_views{T}
    xx::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    xLxL::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    xUxU::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    xy::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    yx::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    yy::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    yalyal::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
end

function H_symmetric_views(H::SparseMatrixCSC,idx::Indices)
    xx = view(H,idx.x,idx.x)
    xLxL = view(H,CartesianIndex.(idx.xL,idx.xL))
    xUxU = view(H,CartesianIndex.(idx.xU,idx.xU))
    xy = view(H,idx.x,idx.y)
    yx = view(H,idx.y,idx.x)
    yy = view(H,CartesianIndex.(idx.y,idx.y))
    yalyal = view(H,CartesianIndex.(idx.y_al,idx.y_al))

    H_symmetric_views(xx,xLxL,xUxU,xy,yx,yy,yalyal)
end

# TODO: shouldn't this be `Base.copyto(x,y)`?
# TODO: should be able to combine these into a single function, at least using `Union`
"""
    update!(x,y)

Copy `y` to `x` element-wise, where `x` is a view into a `SparseMatrixCSC`, and `y` is either
a `SparseMatrixCSC` or a number.
"""
function update!(x::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false},y::SparseMatrixCSC{T,Int}) where T
    x .= y
    return nothing
end
function update!(x::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false},y::Adjoint{T,SparseMatrixCSC{T,Int}}) where T
    x .= y
    return nothing
end
function update!(x::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false},y::T) where T
    x .= y
    return nothing
end
function update!(x::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false},y::Vector{T}) where T
    x .= y
    return nothing
end

"""
    add_update!(x,y)

Add `y` to `x` element-wise, where `x` is a view into a `SparseMatrixCSC`, and `y` is either
a `Vector` or a number.
"""
function add_update!(x::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false},y::Vector{T}) where T
    x .+= y
    return nothing
end
function add_update!(x::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false},y::T) where T
    x .+= y
    return nothing
end
function add_update!(x::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false},y::SubArray{S}) where {T,S}
    x .+= y
    return nothing
end
