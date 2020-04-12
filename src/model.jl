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
    ∇²cλ_func!::Function
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

function constraint_functions(c_func::Function)
    function c_func!(c,x,ρ)
        _c_func(x) = c_func(x,ρ)
        c .= _c_func(x)
        return nothing
    end

    function ∇c_func!(∇c,x)
        _c_func(x) = c_func(x,0.)
        ∇c .= ForwardDiff.jacobian(_c_func,x)
        return nothing
    end

    function ∇²cλ_func!(∇²cλ,x,λ)
        _c_func(x) = c_func(x,0.)
        ∇c_func(x) = ForwardDiff.jacobian(_c_func,x)
        ∇cλ(x) = ∇c_func(x)'*λ
        ∇²cλ .= ForwardDiff.jacobian(∇cλ,x)
        return return nothing
    end

    return c_func!, ∇c_func!, ∇²cλ_func!
end
