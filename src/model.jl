abstract type AbstractModel end

mutable struct Model{T} <: AbstractModel
    n::Int
    m::Int

    xL::Vector{T}
    xU::Vector{T}

    f_func::Function
    ∇f_func!::Function
    ∇²f_func!::Function

    c_func!::Function
    ∇c_func!::Function
    ∇²cy_func!::Function
end

function objective_functions(f::Function)
    function ∇f_func!(∇f,x)
        ∇f .= ForwardDiff.gradient(f_func,x)
    end
    function ∇²f_func!(∇²f,x)
        ∇²f .= ForwardDiff.hessian(f_func,x)
    end
    return f, ∇f_func!, ∇²f_func!
end

function constraint_functions(c::Function)
    function c_func!(c,x)
        c .= c_func(x)
        return nothing
    end

    function ∇c_func!(∇c,x)
        ∇c .= ForwardDiff.jacobian(c_func,x)
        return nothing
    end

    function ∇²cy_func!(∇²cy,x,y)
        ∇c_func(x) = ForwardDiff.jacobian(c_func,x)
        ∇cy(x) = ∇c_func(x)'*y
        ∇²cy .= ForwardDiff.jacobian(∇cy,x)
        return return nothing
    end

    return c_func!, ∇c_func!, ∇²cy_func!
end
