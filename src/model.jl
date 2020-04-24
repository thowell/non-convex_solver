abstract type AbstractModel end
abstract type AbstractModelInfo end
struct EmptyModelInfo <: AbstractModelInfo end

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

function objective_functions(f::Function)
    function f_func(x)
        return f(x)
    end
    function ∇f_func!(∇f,x)
        ∇f .= ForwardDiff.gradient(f,x)
    end
    function ∇²f_func!(∇²f,x)
        ∇²f .= ForwardDiff.hessian(f,x)
    end
    return f_func, ∇f_func!, ∇²f_func!
end

function constraint_functions(c::Function)
    function c_func!(c,x)
        c .= c_func(x)
        return nothing
    end

    function ∇c_func!(∇c,x)
        ∇c .= ForwardDiff.jacobian(c,x)
        return nothing
    end

    function ∇²cy_func!(∇²cy,x,y)
        ∇c_func(x) = ForwardDiff.jacobian(c,x)
        ∇cy(x) = ∇c_func(x)'*y
        ∇²cy .= ForwardDiff.jacobian(∇cy,x)
        return return nothing
    end

    return c_func!, ∇c_func!, ∇²cy_func!
end
