mutable struct Solver{T}
    model::AbstractModel
    model_opt::AbstractModel # optimization model provided to the solver

    x::Vector{T}
    xl::SubArray{T,1,Array{T,1},Tuple{Array{Int,1}},false}
    xu::SubArray{T,1,Array{T,1},Tuple{Array{Int,1}},false}
    xx::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int}},true}
    xs::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int}},true}
    xr::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int}},true}

    x⁺::Vector{T}

    ΔxL::Vector{T}                  # lower bounds error (nL,)
    ΔxU::Vector{T}                  # upper bounds error (nU,)

    y::Vector{T}                    # dual variables (m,)

    zL::Vector{T}                   # duals for lower bound constraint (nL,)
    zU::Vector{T}                   # duals for upper bound constraint (nU,)

    σL::Vector{T}
    σU::Vector{T}

    φ::T                            # barrier objective value
    φ⁺::T                           # next barrier objective value
    ∇φ::Vector{T}                   # gradient of barrier objective

    ∇L::Vector{T}                   # gradient of the Lagrangian
    ∇²L::SparseMatrixCSC{T,Int}     # Hessian of the Lagrangian

    c::Vector{T}                    # constraint values
    c_soc::Vector{T}
    c_tmp::Vector{T}

    H::SparseMatrixCSC{T,Int}       # KKT matrix
    H_sym::SparseMatrixCSC{T,Int}   # Symmetric KKT matrix

    Hv::H_fullspace_views{T}
    Hv_sym::H_symmetric_views{T}

    h::Vector{T}                    # rhs of KKT system
    hx::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int}},true}
    hy
    hzL
    hzU

    h_sym::Vector{T}                # rhs of symmetric KKT system

    linear_solver::LinearSolver

    d::Vector{T}                    # current step
    dx::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int}},true}   # current step in the primals
    dxL::SubArray{T,1,Array{T,1},Tuple{Array{Int,1}},false}   # current step in the primals with lower bounds
    dxU::SubArray{T,1,Array{T,1},Tuple{Array{Int,1}},false}   # current step in the primals with upper bounds
    dy::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int}},true}   # current step in the duals
    dxy::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int}},true}
    dzL::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int}},true}  # current step in the slack duals
    dzU::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int}},true}  # current step in the slack duals

    Δ::Vector{T}    # iterative refinement step
    Δ_xL::SubArray{T,1,Array{T,1},Tuple{Array{Int,1}},false}
    Δ_xU::SubArray{T,1,Array{T,1},Tuple{Array{Int,1}},false}
    Δ_xy::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int}},true}
    Δ_zL::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int}},true}
    Δ_zU::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int}},true}

    res::Vector{T}  # iterative refinement residual
    res_xL::SubArray{T,1,Array{T,1},Tuple{Array{Int,1}},false}
    res_xU::SubArray{T,1,Array{T,1},Tuple{Array{Int,1}},false}
    res_xy::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int}},true}
    res_zL::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int}},true}
    res_zU::SubArray{T,1,Array{T,1},Tuple{UnitRange{Int}},true}

    # Line search values
    step_size::T
    dual_step_size::T
    maximum_step_size::T
    minimum_step_size::T

    # Regularization
    regularization::Vector{T}
    primal_regularization::T
    primal_regularization_last::T
    dual_regularization::T

    # Constraint violation
    constraint_violation::T          # 1-norm of constraint violation
    constraint_violation_candidate::T
    min_constraint_violation::T
    max_constraint_violation::T
    constraint_violation_correction::T

    # Penalty values
    central_path::T
    fraction_to_boundary::T
    filter::Vector{Tuple}

    # iteration counts
    j::Int   # central path iteration (outer loop)
    k::Int   # barrier problem iteration
    l::Int   # line search
    p::Int   # second order corrections
    t::Int

    x_copy::Vector{T}
    y_copy::Vector{T}
    zL_copy::Vector{T}
    zU_copy::Vector{T}
    d_copy::Vector{T}
    d_copy_2::Vector{T}

    Fcentral_path::Vector{T}

    idx::Indices

    fail_cnt::Int

    df::T
    Dc::SparseMatrixCSC{T,Int}

    penalty::T
    dual::Vector{T}

    opts::Options{T}
end

function Solver(x0,model::AbstractModel,model_opt::AbstractModel;opts=Options{Float64}())
    n = model.n
    m = model.m
    mI = model.mI
    mE = model.mE
    mA = model.mA
    cI_idx = model.cI_idx
    cE_idx = model.cE_idx
    cA_idx = model.cA_idx
    xL = model.xL
    xU = model.xU
    xL_bool = model.xL_bool
    xU_bool = model.xU_bool
    xLs_bool = model.xLs_bool
    xUs_bool = model.xUs_bool
    nL = model.nL
    nU = model.nU

    idx = indices(model,model_opt)

    # initialize primals
    x = zeros(n)
    xl = view(x,idx.xL)
    xu = view(x,idx.xU)
    xx = view(x,1:model_opt.n)
    xs = view(x,idx.s)
    xr = view(x,idx.r)

    x⁺ = zeros(n)

    ΔxL = zeros(nL)
    ΔxU = zeros(nU)

    opts.relax_bnds && relax_bounds!(xL,xU,xL_bool,xU_bool,n,opts.residual_tolerance)

    for i = 1:n
        x[i] = initialize_variables(x0[i],xL[i],xU[i],opts.κ1,opts.κ2)
    end

    zL = opts.zL0*ones(nL)
    zU = opts.zU0*ones(nU)

    H = spzeros(n+m+nL+nU,n+m+nL+nU)
    h = zeros(n+m+nL+nU)

    hx = view(h,1:model_opt.n)
    hy = view(h,idx.y)
    hzL = view(h,idx.zL[1:model_opt.nL])
    hzU = view(h,idx.zU)

    H_sym = spzeros(n+m,n+m)
    h_sym = zeros(n+m)

    if opts.linear_solver == :QDLDL
        F = qdldl(sparse(1.0*I,n+m,n+m))
        inertia = Inertia(0,0,0)
        linear_solver = QDLDLSolver(F,inertia)
    else
        @error("linear solver not supported")
    end

    ∇²L = spzeros(n,n)
    σL = zeros(nL)
    σU = zeros(nU)

    eval_∇c!(model,x)
    Dc = constraint_scaling(opts.g_max,get_∇c(model),m)

    central_path = copy(opts.central_path_initial)
    penalty = 1.0
    dual = zeros(mA)
    τ = fraction_to_boundary(central_path,opts.min_fraction_to_boundary)

    eval_∇f!(model,x)
    df = objective_gradient_scaling(opts.g_max,get_∇f(model))

    φ = 0.
    φ⁺ = 0.
    ∇φ = zeros(n)

    ∇L = zeros(n)

    c = zeros(m)
    c_soc = zeros(m)
    c_tmp = zeros(m)

    eval_c!(model,x)
    get_c_scaled!(c,model,Dc,opts.nlp_scaling)

    d = zeros(n+m+nL+nU)
    d_soc = zeros(n+m+nL+nU)

    step_size = 1.0
    dual_step_size = 1.0
    maximum_step_size = 1.0
    minimum_step_size = 1.0

    regularization = zero(d)
    primal_regularization = 0.
    primal_regularization_last = 0.
    dual_regularization = 0.

    y = zeros(m)

    filter = Tuple[]

    j = 0
    k = 0
    l = 0
    p = 0
    t = 0

    x_copy = zeros(n)
    y_copy = zeros(m)
    zL_copy = zeros(nL)
    zU_copy = zeros(nU)
    d_copy = zero(d)
    d_copy_2 = zero(d)

    Fcentral_path = zeros(n+m+nL+nU)

    fail_cnt = 0

    Hv = H_fullspace_views(H,idx)
    Hv_sym = H_symmetric_views(H_sym,idx)

    constraint_violation = norm(c,1)
    constraint_violation_candidate = copy(constraint_violation)
    min_constraint_violation = initialize_min_constraint_violation(constraint_violation)
    max_constraint_violation = initialize_max_constraint_violation(constraint_violation)

    constraint_violation_correction = 0.

    dx = view(d,idx.x)
    dxL = view(d,idx.xL)
    dxU = view(d,idx.xU)
    dy = view(d,idx.y)
    dxy = view(d,idx.xy)
    dzL = view(d,idx.zL)
    dzU = view(d,idx.zU)

    Δ = zero(d)
    Δ_xL = view(Δ,idx.xL)
    Δ_xU = view(Δ,idx.xU)
    Δ_xy = view(Δ,idx.xy)
    Δ_zL = view(Δ,idx.zL)
    Δ_zU = view(Δ,idx.zU)

    res = zero(d)
    res_xL = view(res,idx.xL)
    res_xU = view(res,idx.xU)
    res_xy = view(res,idx.xy)
    res_zL = view(res,idx.zL)
    res_zU = view(res,idx.zU)

    Solver(model,model_opt,
           x,xl,xu,xx,xs,xr,
           x⁺,
           ΔxL,ΔxU,
           y,
           zL,zU,σL,σU,
           φ,φ⁺,∇φ,
           ∇L,∇²L,
           c,c_soc,c_tmp,
           H,H_sym,
           Hv,Hv_sym,
           h,hx,hy,hzL,hzU,
           h_sym,
           linear_solver,
           d,dx,dxL,dxU,dy,dxy,dzL,dzU,
           Δ,Δ_xL,Δ_xU,Δ_xy,Δ_zL,Δ_zU,
           res,res_xL,res_xU,res_xy,res_zL,res_zU,
           step_size,dual_step_size,maximum_step_size,minimum_step_size,
           regularization,primal_regularization,primal_regularization_last,dual_regularization,
           constraint_violation,constraint_violation_candidate,min_constraint_violation,max_constraint_violation,constraint_violation_correction,
           central_path,τ,
           filter,
           j,k,l,p,t,
           x_copy,y_copy,zL_copy,zU_copy,d_copy,d_copy_2,
           Fcentral_path,
           idx,
           fail_cnt,
           df,Dc,
           penalty,dual,
           opts)
end

"""
    tolerance(x, y, zL, zU, ∇xL, ∇xU, c, ∇L, central_path, sd, sc, penalty, dual, yA, cA)
    tolerance(solver::Solver)

Evaluate the optimality error.
"""
function tolerance(zL,zU,ΔxL,ΔxU,c,∇L,central_path)
    return max(norm(∇L,Inf),
               norm(c,Inf),
               norm(ΔxL.*zL .- central_path,Inf),
               norm(ΔxU.*zU .- central_path,Inf))
end

tolerance(central_path,s::Solver) = tolerance(s.zL,s.zU,s.ΔxL,s.ΔxU,s.c,s.∇L,central_path)

"""
    bounds!(s::Solver)

Evaluate the bound constraints and their sigma values
"""
function bounds!(s::Solver)
    s.ΔxL .= s.xl - view(s.model.xL,s.idx.xL)
    s.ΔxU .= view(s.model.xU,s.idx.xU) - s.xu
    s.σL .= s.zL./(s.ΔxL .- s.dual_regularization)
    s.σU .= s.zU./(s.ΔxU .- s.dual_regularization)
    return nothing
end

"""
    eval_objective!(s::Solver)

Evaluate the objective value and it's first and second-order derivatives
"""
function eval_objective!(s::Solver)
    eval_∇f!(s.model,s.x)
    eval_∇²f!(s.model,s.x)
    
    return nothing
end

"""
    eval_constraints!(s::Solver)

Evaluate the constraints and their first and second-order derivatives. Also compute the
constraint residual `constraint_violation`.
"""
function eval_constraints!(s::Solver)
    eval_c!(s.model,s.x)
    get_c_scaled!(s.c,s)

    eval_∇c!(s.model,s.x)

    eval_∇²cy!(s.model,s.x,s.y)
   
    s.constraint_violation = norm(s.c,1)
    return nothing
end

function get_c_scaled!(c,model,Dc,nlp_scaling)
    nlp_scaling && (c .= Dc*get_c(model))
    return nothing
end

function get_c_scaled!(c,s::Solver)
    get_c_scaled!(c,s.model,s.Dc,s.opts.nlp_scaling)
end

"""
    eval_lagrangian!(s::Solver)

Evaluate the first and second derivatives of the Lagrangian
"""
function eval_lagrangian!(s::Solver)
    s.∇L .= get_∇f(s.model)
    s.∇L .+= get_∇c(s.model)'*s.y
    s.∇L[s.idx.xL] -= s.zL
    s.∇L[s.idx.xU] += s.zU
    s.model.mA > 0 && (s.∇L[s.idx.r] += s.dual + s.penalty*view(s.x,s.idx.r))

    s.∇²L .= get_∇²f(s.model) + get_∇²cy(s.model)
    s.model.mA > 0 && (view(s.∇²L,CartesianIndex.(s.idx.r,s.idx.r)) .+= s.penalty)
    return nothing
end

"""
    eval_barrier(s::Solver)

Evaluate barrier objective and it's gradient
"""
function eval_barrier!(s::Solver)
    s.φ = get_f(s,s.x)
    s.φ -= s.central_path*sum(log.(s.ΔxL))
    s.φ -= s.central_path*sum(log.(s.ΔxU))
    s.model.mA > 0 && (s.φ += s.dual'*view(s.x,s.idx.r) + 0.5*s.penalty*view(s.x,s.idx.r)'*view(s.x,s.idx.r))

    s.∇φ .= get_∇f(s.model)
    s.∇φ[s.idx.xL] -= s.central_path./s.ΔxL
    s.∇φ[s.idx.xU] += s.central_path./s.ΔxU
    s.model.mA > 0 && (s.∇φ[s.idx.r] += s.dual + s.penalty*view(s.x,s.idx.r))
    
    return nothing
end

"""
    step!(s::Solver)

Evaluate all critical values for the current iterate stored in `s.x` and `s.y`, including
bound constraints, objective, constraints, Lagrangian, and barrier objective, and their
required derivatives.
"""
function step!(s::Solver)
    bounds!(s)
    eval_objective!(s)
    eval_constraints!(s)
    eval_lagrangian!(s)
    eval_barrier!(s)
    kkt_gradient_fullspace!(s)
    return nothing
end

"""
    central_path(central_path, κcentral_path, θcentral_path, residual_tolerance)
    central_path(s::Solver)

Update the penalty parameter (Eq. 7) with constants κcentral_path ∈ (0,1), θcentral_path ∈ (1,2)
"""
central_path(central_path, κcentral_path, θcentral_path, residual_tolerance) = max(residual_tolerance/10.,min(κcentral_path*central_path,central_path^θcentral_path))
function central_path!(s::Solver)
    s.central_path = central_path(s.central_path, s.opts.κcentral_path, s.opts.θcentral_path, s.opts.residual_tolerance)
    return nothing
end

"""
    fraction_to_boundary(central_path, min_fraction_to_boundary)
    fraction_to_boundary(s::Solver)

Update the "fraction-to-boundary" parameter (Eq. 8) where min_fraction_to_boundary ∈ (0,1) is it's minimum value.
"""
fraction_to_boundary(central_path,min_fraction_to_boundary) = max(min_fraction_to_boundary,1.0-central_path)
function fraction_to_boundary!(s::Solver)
    s.fraction_to_boundary = fraction_to_boundary(s.central_path,s.opts.min_fraction_to_boundary)
    return nothing
end

"""
    fraction_to_boundary(x, d, step_size, fraction_to_boundary)

Check if the `x` satisfies the "fraction-to-boundary" rule (Eq. 15)
"""
fraction_to_boundary(x,d,step_size,fraction_to_boundary) = all(x + step_size*d .>= (1 - fraction_to_boundary)*x)
function fraction_to_boundary_bounds(xl,xL,xu,xU,dxL,dxU,step_size,fraction_to_boundary)
    return all((xU-(xu + step_size*dxU)) .>= (1 - fraction_to_boundary)*(xU-xu)) && all(((xl + step_size*dxL)-xL) .>= (1 - fraction_to_boundary)*(xl-xL))
end

function initialize_max_constraint_violation(constraint_violation)
    max_constraint_violation = 1.0e4*max(1.0,constraint_violation)
    return max_constraint_violation
end

function initialize_min_constraint_violation(constraint_violation)
    min_constraint_violation = 1.0e-4*max(1.0,constraint_violation)
    return min_constraint_violation
end

"""
    initialize_variables(x, xL, xU, κ1, κ2)

Initilize the primal variables with a feasible guess wrt the bound constraints, projecting
the provided guess `x0` slightly inside of the feasible region, with `κ1`, `κ2` ∈ (0,0.5)
determining how far into the interior the value is projected.
"""
function initialize_variables(x,xL,xU,κ1,κ2)
    pl = min(κ1*max(1.0,abs(xL)),κ2*(xU-xL))
    pu = min(κ1*max(1.0,abs(xU)),κ2*(xU-xL))

    # projection
    if x < xL+pl
        x = xL+pl
    elseif x > xU-pu
        x = xU-pu
    end
    return x
end

"""
    constraint_violation(x, s::Solver)

Calculate the 1-norm of the constraints
"""
function constraint_violation(x,s::Solver)
    eval_c!(s.model,x)
    get_c_scaled!(s.c_tmp,s)
    return norm(s.c_tmp,1)
end

"""
    barrier(x, xL, xU, xL_bool, xU_bool, xLs_bool xUs_bool, central_path, κd, f, penalty, yA, cA)
    barrier(x, s::Solver)

Calculate the barrier objective function. When called using the solver, re-calculates the
    objective `f` and the constraints `c`.
"""
function barrier(f,xl,xL,xu,xU,xls,xLs,xus,xUs,central_path,κd,r,dual,penalty)
    return (f
            - central_path*sum(log.(xl - xL)) - central_path*sum(log.(xU - xu))
            + κd*central_path*sum(xls - xLs) + κd*central_path*sum(xUs - xus)
            + dual'*r + 0.5*penalty*r'*r)
end

function barrier(x,s::Solver)
    eval_c!(s.model,x)
    get_c_scaled!(s.c_tmp,s)

    return barrier(get_f(s,x),
                   view(x,s.idx.xL),view(s.model.xL,s.idx.xL),
                   view(x,s.idx.xU),view(s.model.xU,s.idx.xU),
                   view(x,s.idx.xLs),view(s.model.xL,s.idx.xLs),
                   view(x,s.idx.xUs),view(s.model.xU,s.idx.xUs),
                   s.central_path,0.,
                   view(x,s.idx.r),s.dual,s.penalty)
end

"""
    accept_step!(s::Solver)

Accept the current step, copying the candidate primals and duals into the current iterate.
"""
function accept_step!(s::Solver)
    s.x .= s.x⁺
    s.y .+= s.step_size*s.dy
    s.zL .+= s.dual_step_size*s.dzL
    s.zU .+= s.dual_step_size*s.dzU
    return nothing
end

function Solver(x0,model;opts=Options{Float64}()) where T
    if model.mI > 0 || model.mA > 0
        # slack model
        model_s = slack_model(model,bnd_tol=opts.bnd_tol)

        # initialize slacks
        eval_c!(model,x0)
        s0 = get_c(model)[model.cI_idx]
        r0 = get_c(model)[model.cA_idx]
        _x0 = [x0;s0;r0]
    else
        model_s = model
        _x0 = x0
    end
    
    return Solver(_x0,model_s,model,opts=opts)
end

function get_f(s::Solver,x)
    (s.opts.nlp_scaling ? s.df*get_f(s.model,x) : get_f(s.model,x))
end

function get_solution(s::Solver)
    return s.x[1:s.model_opt.n]
end
