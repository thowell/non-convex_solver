abstract type AbstractModelInfo end
abstract type AbstractModel <: AbstractModelInfo end
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

    # data
    ∇f::Vector{T}
    ∇f_prev::Vector{T}
    ∇²f::SparseMatrixCSC{T,Int}

    c::Vector{T}
    ∇c::SparseMatrixCSC{T,Int}
    ∇c_prev::SparseMatrixCSC{T,Int}
    ∇²cy::SparseMatrixCSC{T,Int}

    info::AbstractModelInfo
end

function Model(n,m,xL,xU,f_func,c_func;cI_idx=zeros(Bool,m),cA_idx=ones(Bool,m))
    f, ∇f!, ∇²f! = objective_functions(f_func)
    c!, ∇c!, ∇²cy! = constraint_functions(c_func)
    Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=cI_idx,cA_idx=cA_idx)
end

function Model(n,m,xL,xU,f_func,∇f_func!,∇²f_func!,c_func!,∇c_func!,∇²cy_func!;
        cI_idx=zeros(Bool,m),cA_idx=ones(Bool,m),bnd_tol=1.0e8)

    mI = convert(Int,sum(cI_idx))
    cE_idx = Vector((cI_idx + cA_idx) .== 0)
    mE = convert(Int,sum(cE_idx))
    mA = convert(Int,sum(cA_idx))

    xL_bool, xU_bool, xLs_bool, xUs_bool = bool_bounds(xL,xU,bnd_tol)

    nL = convert(Int,sum(xL_bool))
    nU = convert(Int,sum(xU_bool))

    # data
    ∇f = zeros(n)
    ∇f_prev = zeros(n)
    ∇²f = spzeros(n,n)

    c = zeros(m)
    ∇c = spzeros(m,n)
    ∇c_prev = spzeros(m,n)
    ∇²cy = spzeros(n,n)

    info = EmptyModelInfo()

    Model(n,m,mI,mE,mA,
          xL,xU,xL_bool,xU_bool,xLs_bool,xUs_bool,nL,nU,
          f_func,∇f_func!,∇²f_func!,
          c_func!,∇c_func!,∇²cy_func!,
          cI_idx,cE_idx,cA_idx,
          ∇f,∇f_prev,∇²f,
          c,∇c,∇c_prev,∇²cy,
          info)
end

function eval_∇f!(model::Model,x)
    model.∇f_prev .= model.∇f
    model.∇f_func!(model.∇f,x,model)
    return nothing
end

function eval_∇²f!(model::Model,x)
    model.∇²f_func!(model.∇²f,x,model)
    return return nothing
end

function get_f(model::Model,x)
    model.f_func(x,model)
end

function get_∇f(model::Model)
    return model.∇f
end

function get_∇²f(model::Model)
    return model.∇²f
end

function eval_c!(model::Model,x)
    model.c_func!(model.c,x,model)
    return nothing
end

function eval_∇c!(model::Model,x)
    model.∇c_prev .= model.∇c
    model.∇c_func!(model.∇c,x,model)
    return nothing
end

function eval_∇²cy!(model::Model,x,y)
    model.∇²cy_func!(model.∇²cy,x,y,model)
    return nothing
end

function get_c(model::Model)
    return model.c
end

function get_∇c(model::Model)
    return model.∇c
end

function get_∇²cy(model::Model)
    return model.∇²cy
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

mutable struct SlackModelInfo <: AbstractModelInfo
    model::Model
end

function slack_model(model::Model;bnd_tol=1.0e8)
    # slack bounds
    xL_slack = [zeros(model.mI);-Inf*ones(model.mA)]
    xU_slack = Inf*ones(model.mI+model.mA)
    xL_bool_slack, xU_bool_slack, xLs_bool_slack, xUs_bool_slack = bool_bounds(xL_slack,xU_slack,bnd_tol)

    xL = [model.xL;xL_slack]
    xU = [model.xU;xU_slack]

    xL_bool = [model.xL_bool;xL_bool_slack]
    xU_bool = [model.xU_bool;xU_bool_slack]

    xLs_bool = [model.xLs_bool;xLs_bool_slack]
    xUs_bool = [model.xUs_bool;xUs_bool_slack]

    # dimensions
    n = model.n + model.mI + model.mA
    m = model.m

    nL = convert(Int,sum(xL_bool))
    nU = convert(Int,sum(xU_bool))

    # modified constraint functions
    function f_func(x,model)
        _model = model.info.model
        return _model.f_func(view(x,1:_model.n),_model)
    end
    function ∇f_func!(∇f,x,model)
        _model = model.info.model
        _model.∇f_func!(view(∇f,1:_model.n),view(x,1:_model.n),_model)
        return nothing
    end
    function ∇²f_func!(∇²f,x,model)
        _model = model.info.model
        _model.∇²f_func!(view(∇²f,1:_model.n,1:_model.n),view(x,1:_model.n),_model)
        return nothing
    end

    # modified constraint functions
    function c_func!(c,x,model)
        _model = model.info.model
        _model.c_func!(view(c,1:_model.m),view(x,1:_model.n),_model)
        c[(1:_model.m)[_model.cI_idx]] .-= view(x,_model.n .+ (1:_model.mI))
        c[(1:_model.m)[_model.cA_idx]] .-= view(x,_model.n+_model.mI .+ (1:_model.mA))
        return nothing
    end
    function ∇c_func!(∇c,x,model)
        _model = model.info.model
        _model.∇c_func!(view(∇c,1:_model.m,1:_model.n),view(x,1:_model.n),_model)
        ∇c[CartesianIndex.((1:_model.m)[_model.cI_idx],_model.n .+ (1:_model.mI))] .= -1.0
        ∇c[CartesianIndex.((1:_model.m)[_model.cA_idx],_model.n+_model.mI .+ (1:_model.mA))] .= -1.0
        return nothing
    end
    function ∇²cy_func!(∇²cy,x,y,model)
        _model = model.info.model
        _model.∇²cy_func!(view(∇²cy,1:_model.n,1:_model.n),view(x,1:_model.n),view(y,1:_model.m),_model)
        return nothing
    end

    # data
    ∇f = zeros(n)
    ∇f_prev = zeros(n)
    ∇²f = spzeros(n,n)

    c = zeros(m)
    ∇c = spzeros(m,n)
    ∇c_prev = spzeros(m,n)
    ∇²cy = spzeros(n,n)

    info = SlackModelInfo(model)

    Model(n,m,model.mI,model.mE,model.mA,
          xL,xU,xL_bool,xU_bool,xLs_bool,xUs_bool,nL,nU,
          f_func,∇f_func!,∇²f_func!,
          c_func!,∇c_func!,∇²cy_func!,
          model.cI_idx,model.cE_idx,model.cA_idx,
          ∇f,∇f_prev,∇²f,
          c,∇c,∇c_prev,∇²cy,
          info)
end

mutable struct RestorationModelInfo{T} <: AbstractModelInfo
    model::Model
    xR::Vector{T}
    DR
    ζ::T
    ρ::T
end

function restoration_model(model::Model;bnd_tol=1.0e8)
    # p,n bounds
    xL_pn = zeros(2*model.m)
    xU_pn = Inf*ones(2*model.m)
    xL_bool_pn, xU_bool_pn, xLs_bool_pn, xUs_bool_pn = bool_bounds(xL_pn,xU_pn,bnd_tol)

    xL = [model.xL;xL_pn]
    xU = [model.xU;xU_pn]

    xL_bool = [model.xL_bool;xL_bool_pn]
    xU_bool = [model.xU_bool;xU_bool_pn]

    xLs_bool = [model.xLs_bool;xLs_bool_pn]
    xUs_bool = [model.xUs_bool;xUs_bool_pn]

    # dimensions
    n = model.n + 2*model.m
    m = model.m

    nL = convert(Int,sum(xL_bool))
    nU = convert(Int,sum(xU_bool))

    # restoration objective
    function f_func(x,model)
        _model = model.info.model
        xR = model.info.xR
        DR = model.info.DR
        ζ = model.info.ζ
        ρ = model.info.ρ
        ρ*sum(view(x,_model.n .+ (1:2*_model.m))) + 0.5*ζ*(view(x,1:_model.n) - xR)'*DR'*DR*(view(x,1:_model.n) - xR)
    end

    function ∇f_func!(∇f,x,model)
        _model = model.info.model
        xR = model.info.xR
        DR = model.info.DR
        ζ = model.info.ζ
        ρ = model.info.ρ
        ∇f[1:_model.n] = ζ*DR'*DR*(view(x,1:_model.n) - xR)
        ∇f[_model.n .+ (1:2*_model.m)] .= ρ
        return nothing
    end

    function ∇²f_func!(∇²f,x,model)
        _model = model.info.model
        DR = model.info.DR
        ζ = model.info.ζ
        ∇²f[1:_model.n,1:_model.n] = ζ*DR'*DR
        return nothing
    end

    # modified constraint functions
    function c_func!(c,x,model)
        _model = model.info.model
        _model.c_func!(c,view(x,1:_model.n),_model)
        c .-= view(x,_model.n .+ (1:_model.m))
        c .+= view(x,_model.n+_model.m .+ (1:_model.m))
        return nothing
    end
    function ∇c_func!(∇c,x,model)
        _model = model.info.model
        _model.∇c_func!(view(∇c,1:_model.m,1:_model.n),view(x,1:_model.n),_model)
        view(∇c,CartesianIndex.(1:_model.m,_model.n .+ (1:_model.m))) .= -1.0
        view(∇c,CartesianIndex.(1:_model.m,_model.n+_model.m .+ (1:_model.m))) .= 1.0
        return nothing
    end
    function ∇²cy_func!(∇²cy,x,y,model)
        _model = model.info.model
        _model.∇²cy_func!(view(∇²cy,1:_model.n,1:_model.n),view(x,1:_model.n),view(y,1:_model.m),_model)
        return nothing
    end

    # data
    ∇f = zeros(n)
    ∇f_prev = zeros(n)
    ∇²f = spzeros(n,n)

    c = zeros(m)
    ∇c = spzeros(m,n)
    ∇c_prev = spzeros(m,n)
    ∇²cy = spzeros(n,n)

    info = RestorationModelInfo(model,zeros(model.n),spzeros(model.n,model.n),0.,0.)

    Model(n,m,model.mI,model.mE,model.mA,
          xL,xU,xL_bool,xU_bool,xLs_bool,xUs_bool,nL,nU,
          f_func,∇f_func!,∇²f_func!,
          c_func!,∇c_func!,∇²cy_func!,
          model.cI_idx,model.cE_idx,model.cA_idx,
          ∇f,∇f_prev,∇²f,
          c,∇c,∇c_prev,∇²cy,
          info)
end
