using BenchmarkTools

n = 1000
m = 10
H = spzeros(n,n)
A = rand(m,m)
v = view(H,1:m,1:m)

h = zeros(n)
idx = 1:m
y = rand(m)
typeof(idx)
function f1(h::Vector{T},y::Vector{T},idx::UnitRange{Int}) where T
    for i = 1:1000
        h[idx] = y
    end
    return nothing
end
function func1(H::SparseMatrixCSC{T,Int},A::Array{T,2}) where T
    for i = 1:1000
        H[1:m,1:m] .= A
    end
    return nothing
end
function func2(H::SparseMatrixCSC{T,Int},A::Array{T,2}) where T
    for i = 1:1000
        view(H,1:m,1:m) .= A
    end
    return nothing
end
function func3(v::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false},A::Array{T,2}) where T
    for i = 1:1000
        v .= A
    end
    return nothing
end
@benchmark f1($h,$y,$idx)
@benchmark func1($H,$A)
@benchmark func2($H,$A)
@benchmark func3($v,$A)

using BenchmarkTools

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
#

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
    ∇f::Vector{T}
    ∇²f::SparseMatrixCSC{T,Int}

    c::Vector{T}
    ∇c::SparseMatrixCSC{T,Int}
    ∇²cy::SparseMatrixCSC{T,Int}
end

function Model(n,m,xL,xU,f_func,∇f_func!,∇²f_func!,c_func!,∇c_func!,∇²cy_func!)
    ∇f = zeros(n)
    ∇²f = spzeros(n,n)

    c = zeros(m)
    ∇c = spzeros(m,n)
    ∇²cy = spzeros(n,n)

    Model(n,m,xL,xU,f_func,∇f_func!,∇²f_func!,c_func!,∇c_func!,∇²cy_func!,∇f,∇²f,c,∇c,∇²cy)
end

f_func(x) = x'*x
function ∇f_func!(model::AbstractModel,x)
    model.∇f .= 2*x
    return nothing
end
function ∇²f_func!(model::AbstractModel,x)
    model.∇²f .= 2.0
    return nothing
end

function c_func!(model::AbstractModel,x)
    model.c .= x[1:m].^2 .- 1.2
    return nothing
end

function ∇c_func!(model::AbstractModel,x)
    model.∇c[:,1:m] = 2.0*Diagonal(x[1:m])
    return nothing
end
function ∇²cy_func!(model::AbstractModel,x,y)
    model.∇²c[1:m,1:m] = 2.0*Diagonal(y)
    return nothing
end

model = Model(n,m,xL,xU,f_func,∇f_func!,∇²f_func!,c_func!,∇c_func!,∇²cy_func!)

@benchmark model.c_func!($model,$s.s.x)
