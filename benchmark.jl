using BenchmarkTools, SparseArrays, LinearAlgebra

n = 1000
m = 10
H = spzeros(n,n)
A = rand(m,m)
v = view(H,1:m,1:m)

# h = zeros(n)
# idx = 1:m
# y = rand(m)
# typeof(idx)
# function f1(h::Vector{T},y::Vector{T},idx::UnitRange{Int}) where T
#     for i = 1:1000
#         h[idx] = y
#     end
#     return nothing
# end
# function func1(H::SparseMatrixCSC{T,Int},A::Array{T,2}) where T
#     for i = 1:1000
#         H[1:m,1:m] .= A
#     end
#     return nothing
# end
# function func2(H::SparseMatrixCSC{T,Int},A::Array{T,2}) where T
#     for i = 1:1000
#         view(H,1:m,1:m) .= A
#     end
#     return nothing
# end
# function func3(v::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false},A::Array{T,2}) where T
#     for i = 1:1000
#         v .= A
#     end
#     return nothing
# end
# @benchmark f1($h,$y,$idx)
# @benchmark func1($H,$A)
# @benchmark func2($H,$A)
# @benchmark func3($v,$A)
#
# using BenchmarkTools

# @benchmark c_func!($s.s.c,$s.s.x,$idx)
# @benchmark ∇c_func!($s.s.∇c,$s.s.x)
#
# @benchmark ∇²cy_func!($s.s.∇²cy,$s.s.x,$s.s.y)

# v = view(s.s.∇²cy,CartesianIndex.(1:m,1:m))
# function ∇²cy_func2!(v,x,y)
#     v .= y
#     v .*= 2.0
#     return nothing
# end
# @benchmark ∇²cy_func2!($v,$s.s.x,$s.s.y)
#
# function ∇²cy_func3!(v,x,y)
#     v .= (2.0*y)
#     return nothing
# end
# @benchmark ∇²cy_func3!($v,$s.s.x,$s.s.y)
#
function update!(x::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false},y::Vector{T}) where T
    x .= y
    return nothing
end

abstract type AbstractModel end
abstract type AbstractModelInfo end
abstract type AbstractIndices end
abstract type AbstractViews end

struct EmptyModelInfo <: AbstractModelInfo end
struct EmptyIndices <: AbstractIndices end
struct EmptyViews <: AbstractViews end

nc = spzeros(m,n)
typeof(view(nc,CartesianIndex.(1:m,1:m)))

struct ProblemViews{T} <:AbstractViews
    ∇c::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
    ∇²cy::SubArray{T,1,SparseMatrixCSC{T,Int},Tuple{Array{CartesianIndex{2},1}},false}
end

struct ProblemIndices <: AbstractIndices
    c::UnitRange{Int}
end

mutable struct Model{T} <: AbstractModel
    # dimensions
    n::Int
    m::Int

    # primal bounds
    xL::Vector{T}
    xU::Vector{T}

    # objective
    f_func::Function
    ∇f_func!::Function
    ∇²f_func!::Function

    # constraints
    c_func!::Function
    ∇c_func!::Function
    ∇²cy_func!::Function

    # data
    x::Vector{T}
    y::Vector{T}

    f::T
    ∇f::Vector{T}
    ∇²f::SparseMatrixCSC{T,Int}

    c::Vector{T}
    ∇c::SparseMatrixCSC{T,Int}
    ∇²cy::SparseMatrixCSC{T,Int}

    info::AbstractModelInfo
end

function Model(n,m,xL,xU,f_func,∇f_func!,∇²f_func!,c_func!,∇c_func!,∇²cy_func!)
    x = zeros(n)
    y = zeros(m)

    f = 0.0
    ∇f = zeros(n)
    ∇²f = spzeros(n,n)

    c = zeros(m)
    ∇c = spzeros(m,n)
    ∇²cy = spzeros(n,n)

    info = EmptyModelInfo()
    Model(n,m,xL,xU,f_func,∇f_func!,∇²f_func!,c_func!,∇c_func!,∇²cy_func!,x,y,f,∇f,∇²f,c,∇c,∇²cy,info)
end

f_func(x) = x'*x
function ∇f_func!(model::AbstractModel)
    model.∇f .= model.x
    return nothing
end
function ∇²f_func!(model::AbstractModel)
    model.∇²f .= 1.0
    return nothing
end

function c_func!(model::AbstractModel)
    model.c .= 0.5*model.x[1:model.m].^2 .- 1.2
    return nothing
end

function ∇c_func!(model::AbstractModel)
    update!(view(model.∇c,CartesianIndex.(1:model.m,1:model.m)),model.x[1:model.m])
    return nothing
end
function ∇²cy_func!(model::AbstractModel)
    update!(view(model.∇²cy,CartesianIndex.(1:model.m,1:model.m)),model.y)
    return nothing
end

xL = -Inf*ones(n)
xU = Inf*ones(n)
y = ones(m)
model = Model(n,m,xL,xU,f_func,∇f_func!,∇²f_func!,c_func!,∇c_func!,∇²cy_func!)

@benchmark model.∇c_func!($model)
@benchmark model.∇²cy_func!($model)

typeof(model.v.∇c)
typeof(xL[model.idx.c])

model.∇c


solver_∇c = model.∇c

solver_∇c
model.∇c_func!(model,rand(n))
model.∇c[1,1] = 100.

solver_∇c
