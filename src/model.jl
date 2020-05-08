abstract type AbstractModel end
abstract type AbstractModelInfo end
struct EmptyModelInfo <: AbstractModelInfo end

function eval_f(model::AbstractModel, x)
    throw(TO.NotImplemented(:eval_f, model))
end

function grad_f!(model::AbstractModel, x, ∇f)
    f(x) = eval_f(model, x)
    ForwardDiff.gradient!(∇f, f, x)
end

function hess_f!(model::AbstractModel, x, ∇²f)
    f(x) = eval_f(model, x)
    ForwardDiff.hessian!(∇²f, f, x)
end

function eval_c!(model::AbstractModel, x, c)
    throw(TO.NotImplemented(:eval_c, model))
end

function jac_c!(model::AbstractModel, x, c, ∇c)
    c!(c, x) = eval_c!(model, x, c)
    ForwardDiff.jacobian!(∇c, c!, c, x)
end

function hess_cy!(model::AbstractModel, x, y, ∇²cy)
    n,m = size(model)
    jvp(x) = jac_c!(model, x, zeros(eltype(x), m), spzeros(eltype(x), m, n))'y
    ForwardDiff.jacobian!(∇²cy, jvp, x)
end

"""
    Model{T}

Description of the optimization problem being solved, including the objective function `f`,
the m-dimensional constraint function `c`, as well as their first and second-order derivatives.
Also contains values for the primal and dual variables and derivatives of objective and
constraint functions.
"""
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

"""
    objective_functions(f::Function)

Generate the first and second-order derivatives of the scalar-valued function `f` using
`ForwardDiff`. Generates function with the following signatures:
    f_func(       x, model::AbstractModel)
    ∇f_func!(∇f,  x, model::AbstractModel)
    ∇²f_fun!(∇²f, x, model::AbstractModel)

where `x` is the current vector of primal variables and `∇f` and `∇²f` are vectors/matrices
of the appropriate dimension.
"""
function objective_functions(f::Function)
    function f_func(x,model::AbstractModel)
        return f(x)
    end
    function ∇f_func!(∇f,x,model::AbstractModel)
        ForwardDiff.gradient!(∇f,f,x)
    end
    function ∇²f_func!(∇²f,x,model::AbstractModel)
        ForwardDiff.hessian!(∇²f,f,x)
    end
    return f_func, ∇f_func!, ∇²f_func!
end

"""
    constraint_functions(c::Function)

Generate the first and second-order derivatives of the vector-valued function `c` using
`ForwardDiff`. Generates function with the following signatures:
    c_func(_c, x, model::AbstractModel)
    ∇c_func!(∇c, x, model::AbstractModel)
    ∇²cy_fun!(∇²cy, x, y, model::AbstractModel)

where `x` is the current vector of primal variables, `y` is the vector of dual variables,
and `size(∇c) = (m,n)`, `size(∇²cy) = (n,n)`
"""
function constraint_functions(c::Function)
    function c_func!(_c,x,model::AbstractModel)
        _c .= c(x)
        return nothing
    end

    function ∇c_func!(∇c,x,model::AbstractModel)
        ForwardDiff.jacobian!(∇c,c,x)
        return nothing
    end

    function ∇²cy_func!(∇²cy,x,y,model::AbstractModel)
        ∇c_func(x) = ForwardDiff.jacobian(c,x)
        ∇cy(x) = ∇c_func(x)'*y
        ForwardDiff.jacobian!(∇²cy,∇cy,x)
        return return nothing
    end

    return c_func!, ∇c_func!, ∇²cy_func!
end
