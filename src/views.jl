"""
    H_symmetric_views{T}

Views into the fullspace KKT matrix
"""
struct H_fullspace_views{T}
    xx::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    xy::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    yx::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    xLzL::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    zLxL::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    zLzL::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
end

function H_fullspace_views(H::SparseMatrixCSC,idx::Indices)
    xx = view(H,idx.x,idx.x)
    xy = view(H,idx.x,idx.y)
    yx = view(H,idx.y,idx.x)
    xLzL = view(H,CartesianIndex.(idx.xL,idx.zL))
    zLxL = view(H,CartesianIndex.(idx.zL,idx.xL))
    zLzL = view(H,CartesianIndex.(idx.zL,idx.zL))

    H_fullspace_views(xx,xy,yx,xLzL,zLxL,zLzL)
end

"""
    H_symmetric_views{T}

Views into the symmetric KKT matrix
"""
struct H_symmetric_views{T}
    xx::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    xLxL::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    xy::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    yx::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    yy::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
end

function H_symmetric_views(H::SparseMatrixCSC,idx::Indices)
    xx = view(H,idx.x,idx.x)
    xLxL = view(H,CartesianIndex.(idx.xL,idx.xL))
    xy = view(H,idx.x,idx.y)
    yx = view(H,idx.y,idx.x)
    yy = view(H,CartesianIndex.(idx.y,idx.y))

    H_symmetric_views(xx,xLxL,xy,yx,yy)
end

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
