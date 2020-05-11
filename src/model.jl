abstract type AbstractModel end
abstract type AbstractModelInfo end
struct EmptyModelInfo <: AbstractModelInfo end

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
    mI::Int
    mE::Int
    mA::Int

    # primal bounds
    xL::Vector{T}
    xU::Vector{T}

    xL_bool::Vector{Bool}
    xU_bool::Vector{Bool}
    xLs_bool::Vector{Bool}
    xUs_bool::Vector{Bool}

    nL::Int
    nU::Int

    # objective
    f_func::Function
    ∇f_func!::Function
    ∇²f_func!::Function

    # constraints
    c_func!::Function
    ∇c_func!::Function
    ∇²cy_func!::Function

    cI_idx::Vector{Bool}
    cE_idx::Vector{Bool}
    cA_idx::Vector{Bool}

    info::AbstractModelInfo
end

function Model(n,m,xL,xU,f_func,∇f_func!,∇²f_func!,c_func!,∇c_func!,∇²cy_func!;
        cI_idx=zeros(Bool,m),cA_idx=zeros(Bool,m))

    mI = convert(Int,sum(cI_idx))
    cE_idx = Vector((cI_idx + cA_idx) .== 0)
    mE = convert(Int,sum(cE_idx))
    mA = convert(Int,sum(cA_idx))

    xL_bool, xU_bool, xLs_bool, xUs_bool = bool_bounds(xL,xU,1.0e8)

    nL = convert(Int,sum(xL .> -Inf))
    nU = convert(Int,sum(xU .< Inf))

    info = EmptyModelInfo()
    Model(n,m,mI,mE,mA,
          xL,xU,xL_bool,xU_bool,xLs_bool,xUs_bool,nL,nU,
          f_func,∇f_func!,∇²f_func!,
          c_func!,∇c_func!,∇²cy_func!,
          cI_idx,cE_idx,cA_idx,
          info)
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
