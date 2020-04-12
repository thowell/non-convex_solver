struct H_views{T}
    xx::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    xλ::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    λx::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    xLzL::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    zLxL::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    xUzU::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    zUxU::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    zLzL::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    zUzU::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
end

function H_views(H::SparseMatrixCSC,idx::Indices)
    xx = view(H,idx.x,idx.x)
    xλ = view(H,idx.x,idx.λ)
    λx = view(H,idx.λ,idx.x)
    xLzL = view(H,CartesianIndex.(idx.xL,idx.zL))
    zLxL = view(H,CartesianIndex.(idx.zL,idx.xL))
    xUzU = view(H,CartesianIndex.(idx.xU,idx.zU))
    zUxU = view(H,CartesianIndex.(idx.zU,idx.xU))
    zLzL = view(H,CartesianIndex.(idx.zL,idx.zL))
    zUzU = view(H,CartesianIndex.(idx.zU,idx.zU))

    H_views(xx,xλ,λx,xLzL,zLxL,xUzU,zUxU,zLzL,zUzU)
end
