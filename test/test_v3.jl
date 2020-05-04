include("../src/interior_point.jl")


n = 500
m = 100

x0 = ones(n)

xL = -Inf*ones(n)
xL[1] = -10.
xL[2] = -5.
xU = Inf*ones(n)
xU[5] = 20.

f_func(x::Vector{T},model::AbstractModel) where T = x'*x
function ∇f_func!(∇f::Vector{T},x::Vector{T},model::AbstractModel) where T
    ∇f .= 2.0*x
    return nothing
end
function ∇²f_func!(∇²f::SparseMatrixCSC{T,Int},x::Vector{T},model::AbstractModel) where T
    update!(model.info.∇²f,2.0)
    ∇²f .= model.∇²f
    return nothing
end

function c_func!(c::Vector{T},x::Vector{T},model::AbstractModel) where T
    c .= x[model.info.idx].^2 .- 1.2
    return nothing
end
function ∇c_func!(∇c::SparseMatrixCSC{T,Int},x::Vector{T},model::AbstractModel) where T
    update!(model.info.∇c,2.0*x[model.info.idx])
    ∇c .= model.∇c
    return nothing
end
function ∇²cy_func!(∇²cy::SparseMatrixCSC{T,Int},x::Vector{T},y::Vector{T},model::AbstractModel) where T
    update!(model.info.∇²cy,2.0*y)
    ∇²cy .= model.∇²cy
    return nothing
end

model = Model(n,m,xL,xU,f_func,∇f_func!,∇²f_func!,c_func!,∇c_func!,∇²cy_func!)

struct CustomModelInfo{T} <: AbstractModelInfo
    ∇²f::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    ∇c::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    ∇²cy::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    idx::UnitRange{Int}
end

model.info = CustomModelInfo(view(model.∇²f,CartesianIndex.(1:n,1:n)),view(model.∇c,CartesianIndex.(1:m,1:m)),view(model.∇²cy,CartesianIndex.(1:m,1:m)),1:m)

opts = Options{Float64}(kkt_solve=:symmetric,
                        iterative_refinement=true,
                        verbose=false,
                        max_iterative_refinement=100)

s = InteriorPointSolver(x0,model,opts=opts)
# @time solve!(s)

# eval_step!(s.s)
# initialize_restoration_solver!(s.s̄,s.s)
# @time solve_restoration!(s.s̄,s.s,verbose=true)
