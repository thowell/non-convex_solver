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
    α::T
    αz::T
    α_max::T
    α_min::T
    β::T

    # Regularization
    δ::Vector{T}
    δw::T
    δw_last::T
    δc::T

    # Constraint violation
    θ::T          # 1-norm of constraint violation
    θ⁺::T
    θ_min::T
    θ_max::T
    θ_soc::T

    # Scaling factors
    sd::T
    sc::T

    # Penalty values
    μ::T
    τ::T
    filter::Vector{Tuple}

    # iteration counts
    j::Int   # central path iteration (outer loop)
    k::Int   # barrier problem iteration
    l::Int   # line search
    p::Int   # second order corrections
    t::Int
    small_search_direction_cnt::Int

    restoration::Bool
    DR::SparseMatrixCSC{T,Int}  # QUESTION: isn't this Diagonal?

    x_copy::Vector{T}
    y_copy::Vector{T}
    zL_copy::Vector{T}
    zU_copy::Vector{T}
    d_copy::Vector{T}
    d_copy_2::Vector{T}

    Fμ::Vector{T}

    idx::Indices
    idx_r::RestorationIndices

    fail_cnt::Int

    df::T
    Dc::SparseMatrixCSC{T,Int}

    ρ::T
    λ::Vector{T}

    qn::QuasiNewton

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
    idx_r = restoration_indices()

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

    opts.relax_bnds && relax_bounds_init!(xL,xU,xL_bool,xU_bool,n,opts.ϵ_tol)

    for i = 1:n
        x[i] = init_x0(x0[i],xL[i],xU[i],opts.κ1,opts.κ2)
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

    if opts.linear_solver == :MA57
        LBL = Ma57(H_sym)
        inertia = Inertia(0,0,0)
        linear_solver = MA57Solver(LBL,inertia)
    elseif opts.linear_solver == :QDLDL
        F = qdldl(sparse(1.0*I,n+m,n+m))
        inertia = Inertia(0,0,0)
        linear_solver = QDLDLSolver(F,inertia)
    elseif opts.linear_solver == :PARDISO
        ps = PardisoSolver()
        A_pardiso = copy(H_sym)
        inertia = Inertia(0,0,0)
        linear_solver = PARDISOSolver(ps,A_pardiso,inertia)
    end

    ∇²L = spzeros(n,n)
    σL = zeros(nL)
    σU = zeros(nU)

    eval_∇c!(model,x)
    Dc = init_Dc(opts.g_max,get_∇c(model),m)

    μ = copy(opts.μ0)
    ρ = 1.0#1.0/μ
    λ = zeros(mA)
    τ = update_τ(μ,opts.τ_min)

    eval_∇f!(model,x)
    # model.∇f[idx.r] += λ + ρ*view(x,idx.r)
    df = init_df(opts.g_max,get_∇f(model))

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

    α = 1.0
    αz = 1.0
    α_max = 1.0
    α_min = 1.0
    α_soc = 1.0
    β = 1.0

    δ = zero(d)
    δw = 0.
    δw_last = 0.
    δc = 0.

    y = zeros(m)

    opts.y_init_ls ? init_y!(y,H_sym,h_sym,d,zL,zU,get_∇f(model),get_∇c(model),n,m,xL_bool,xU_bool,opts.y_max,linear_solver) : zeros(m)

    sd = init_sd(y,[zL;zU],n,m,opts.s_max)
    sc = init_sc([zL;zU],n,opts.s_max)

    filter = Tuple[]

    j = 0
    k = 0
    l = 0
    p = 0
    t = 0

    small_search_direction_cnt = 0

    restoration = false
    DR = spzeros(0,0)

    x_copy = zeros(n)
    y_copy = zeros(m)
    zL_copy = zeros(nL)
    zU_copy = zeros(nU)
    d_copy = zero(d)
    d_copy_2 = zero(d)

    Fμ = zeros(n+m+nL+nU)

    fail_cnt = 0

    Hv = H_fullspace_views(H,idx)
    Hv_sym = H_symmetric_views(H_sym,idx)

    θ = norm(c,1)
    θ⁺ = copy(θ)
    θ_min = init_θ_min(θ)
    θ_max = init_θ_max(θ)

    θ_soc = 0.

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

    if opts.quasi_newton == :lbfgs
        qn = LBFGS(n=model.n,m=model.m,k=opts.lbfgs_length)
    else
        qn = BFGS(n=model.n,m=model.m)
    end

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
           α,αz,α_max,α_min,β,
           δ,δw,δw_last,δc,
           θ,θ⁺,θ_min,θ_max,θ_soc,
           sd,sc,
           μ,τ,
           filter,
           j,k,l,p,t,small_search_direction_cnt,
           restoration,DR,
           x_copy,y_copy,zL_copy,zU_copy,d_copy,d_copy_2,
           Fμ,
           idx,idx_r,
           fail_cnt,
           df,Dc,
           ρ,λ,
           qn,
           opts)
end

"""
    eval_Eμ(x, y, zL, zU, ∇xL, ∇xU, c, ∇L, μ, sd, sc, ρ, λ, yA, cA)
    eval_Eμ(solver::Solver)

Evaluate the optimality error.
"""
function eval_Eμ(zL,zU,ΔxL,ΔxU,c,∇L,μ,sd,sc)
    return max(norm(∇L,Inf)/sd,
               norm(c,Inf),
               norm(ΔxL.*zL .- μ,Inf)/sc,
               norm(ΔxU.*zU .- μ,Inf)/sc)
end

eval_Eμ(μ,s::Solver) = eval_Eμ(s.zL,s.zU,s.ΔxL,s.ΔxU,s.c,s.∇L,μ,s.sd,s.sc)

"""
    eval_bounds!(s::Solver)

Evaluate the bound constraints and their sigma values
"""
function eval_bounds!(s::Solver)
    s.ΔxL .= s.xl - view(s.model.xL,s.idx.xL)
    s.ΔxU .= view(s.model.xU,s.idx.xU) - s.xu
    s.σL .= s.zL./(s.ΔxL .- s.δc)
    s.σU .= s.zU./(s.ΔxU .- s.δc)
    return nothing
end

"""
    eval_objective!(s::Solver)

Evaluate the objective value and it's first and second-order derivatives
"""
function eval_objective!(s::Solver)
    eval_∇f!(s.model,s.x)

    if s.opts.quasi_newton == :none
        eval_∇²f!(s.model,s.x)
    else
        if s.opts.quasi_newton_approx == :constraints
            eval_∇²f!(s.model,s.x)
        end
    end
    return nothing
end

"""
    eval_constraints!(s::Solver)

Evaluate the constraints and their first and second-order derivatives. Also compute the
constraint residual `θ`.
"""
function eval_constraints!(s::Solver)
    eval_c!(s.model,s.x)
    get_c_scaled!(s.c,s)

    eval_∇c!(s.model,s.x)

    if s.opts.quasi_newton == :none
        eval_∇²cy!(s.model,s.x,s.y)
    else
        if s.opts.quasi_newton == :objective
            eval_∇²cy!(s.model,s.x,s.y)
        end
    end
    s.θ = norm(s.c,1)
    return nothing
end

function get_c_scaled!(c,model,Dc,nlp_scaling)
    nlp_scaling && c .= Dc*get_c(model)
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
    s.model.mA > 0 && (s.∇L[s.idx.r] += s.λ + s.ρ*view(s.x,s.idx.r))

    # damping
    if s.opts.single_bnds_damping
        κd = s.opts.κd
        μ = s.μ
        s.∇L[s.idx.xLs] .+= κd*μ
        s.∇L[s.idx.xUs] .-= κd*μ
    end

    if s.opts.quasi_newton == :none
        s.∇²L .= get_∇²f(s.model) + get_∇²cy(s.model)
        s.model.mA > 0 && (view(s.∇²L,CartesianIndex.(s.idx.r,s.idx.r)) .+= s.ρ)

    end
    return nothing
end

"""
    eval_barrier(s::Solver)

Evaluate barrier objective and it's gradient
"""
function eval_barrier!(s::Solver)
    s.φ = get_f(s,s.x)
    s.φ -= s.μ*sum(log.(s.ΔxL))
    s.φ -= s.μ*sum(log.(s.ΔxU))
    s.model.mA > 0 && (s.φ += s.λ'*view(s.x,s.idx.r) + 0.5*s.ρ*view(s.x,s.idx.r)'*view(s.x,s.idx.r))

    s.∇φ .= get_∇f(s.model)
    s.∇φ[s.idx.xL] -= s.μ./s.ΔxL
    s.∇φ[s.idx.xU] += s.μ./s.ΔxU
    s.model.mA > 0 && (s.∇φ[s.idx.r] += s.λ + s.ρ*view(s.x,s.idx.r))


    # damping
    if s.opts.single_bnds_damping
        κd = s.opts.κd
        μ = s.μ

        s.φ += κd*μ*sum(view(s.x,s.idx.xLs) - view(s.model.xL,s.idx.xLs))
        s.φ += κd*μ*sum(view(s.model.xU,s.idx.xUs) - view(s.x,s.idx.xUs))
        s.∇φ[s.idx.xLs] .+= κd*μ
        s.∇φ[s.idx.xUs] .-= κd*μ
    end
    return nothing
end

"""
    eval_step!(s::Solver)

Evaluate all critical values for the current iterate stored in `s.x` and `s.y`, including
bound constraints, objective, constraints, Lagrangian, and barrier objective, and their
required derivatives.
"""
function eval_step!(s::Solver)
    eval_bounds!(s)
    eval_objective!(s)
    eval_constraints!(s)
    eval_lagrangian!(s)
    eval_barrier!(s)
    kkt_gradient_fullspace!(s)
    return nothing
end

"""
    update_μ(μ, κμ, θμ, ϵ_tol)
    update_μ(s::Solver)

Update the penalty parameter (Eq. 7) with constants κμ ∈ (0,1), θμ ∈ (1,2)
"""
update_μ(μ, κμ, θμ, ϵ_tol) = max(ϵ_tol/10.,min(κμ*μ,μ^θμ))
function update_μ!(s::Solver)
    s.μ = update_μ(s.μ, s.opts.κμ, s.opts.θμ, s.opts.ϵ_tol)
    return nothing
end

"""
    update_τ(μ, τ_min)
    update_τ(s::Solver)

Update the "fraction-to-boundary" parameter (Eq. 8) where τ_min ∈ (0,1) is it's minimum value.
"""
update_τ(μ,τ_min) = max(τ_min,1.0-μ)
function update_τ!(s::Solver)
    s.τ = update_τ(s.μ,s.opts.τ_min)
    return nothing
end

"""
    fraction_to_boundary(x, d, α, τ)

Check if the `x` satisfies the "fraction-to-boundary" rule (Eq. 15)
"""
fraction_to_boundary(x,d,α,τ) = all(x + α*d .>= (1 - τ)*x)
function fraction_to_boundary_bnds(xl,xL,xu,xU,dxL,dxU,α,τ)
    return all((xU-(xu + α*dxU)) .>= (1 - τ)*(xU-xu)) && all(((xl + α*dxL)-xL) .>= (1 - τ)*(xl-xL))
end

"""
    reset_z(z, x, μ, κΣ)
    reset_z(s::Solver)

Reset the bound duals `z` according to (Eq. 16) to ensure global convergence, where `κΣ` is
some constant > 1, usually very large (e.g. 10^10).
"""
reset_z(z,x,μ,κΣ) = max(min(z,κΣ*μ/x),μ/(κΣ*x))

function reset_z!(s::Solver)
    s.ΔxL .= s.xl - view(s.model.xL,s.idx.xL)
    s.ΔxU .= view(s.model.xU,s.idx.xU) - s.xu

    for i = 1:s.model.nL
        s.zL[i] = reset_z(s.zL[i],s.ΔxL[i],s.μ,s.opts.κΣ)
    end

    for i = 1:s.model.nU
        s.zU[i] = reset_z(s.zU[i],s.ΔxU[i],s.μ,s.opts.κΣ)
    end
    return nothing
end

function init_θ_max(θ)
    θ_max = 1.0e4*max(1.0,θ)
    return θ_max
end

function init_θ_min(θ)
    θ_min = 1.0e-4*max(1.0,θ)
    return θ_min
end

"""
    init_x0(x, xL, xU, κ1, κ2)

Initilize the primal variables with a feasible guess wrt the bound constraints, projecting
the provided guess `x0` slightly inside of the feasible region, with `κ1`, `κ2` ∈ (0,0.5)
determining how far into the interior the value is projected.
"""
function init_x0(x,xL,xU,κ1,κ2)
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
    init_y!

Solve for the initial dual variables for the equality constraints (Eq. 36)
"""
function init_y!(y,H,h,d,zL,zU,∇f,∇c,n,m,xL_bool,xU_bool,y_max,linear_solver)

    if m > 0
        H[CartesianIndex.((1:n),(1:n))] .= 1.0
        H[1:n,n .+ (1:m)] .= ∇c'
        H[n .+ (1:m),1:n] .= ∇c

        h[1:n] = ∇f
        h[(1:n)[xL_bool]] -= zL
        h[(1:n)[xU_bool]] += zU
        h[n+1:end] .= 0.


        δ = zeros(n+m)
        δw, δc = regularization_init(linear_solver)
        δ[1:n] .= δw
        δ[n .+ (1:m)] .= -δc

        factorize!(linear_solver,H + Diagonal(δ))
        solve!(linear_solver,view(d,1:(n+m)),-h)
        y .= d[n .+ (1:m)]

        if norm(y,Inf) > y_max || any(isnan.(y))
            @warn "least-squares y init failure:\n y_max = $(norm(y,Inf))"
            y .= 0.
        end
    else
        y .= 0.
    end
    H .= 0.
    return nothing
end

"""
    θ(x, s::Solver)

Calculate the 1-norm of the constraints
"""
function θ(x,s::Solver)
    eval_c!(s.model,x)
    get_c_scaled!(s.c_tmp,s)
    return norm(s.c_tmp,1)
end

"""
    barrier(x, xL, xU, xL_bool, xU_bool, xLs_bool xUs_bool, μ, κd, f, ρ, yA, cA)
    barrier(x, s::Solver)

Calculate the barrier objective function. When called using the solver, re-calculates the
    objective `f` and the constraints `c`.
"""
function barrier(f,xl,xL,xu,xU,xls,xLs,xus,xUs,μ,κd,r,λ,ρ)
    return (f
            - μ*sum(log.(xl - xL)) - μ*sum(log.(xU - xu))
            + κd*μ*sum(xls - xLs) + κd*μ*sum(xUs - xus)
            + λ'*r + 0.5*ρ*r'*r)
end

function barrier(x,s::Solver)
    eval_c!(s.model,x)
    get_c_scaled!(s.c_tmp,s)

    return barrier(get_f(s,x),
                   view(x,s.idx.xL),view(s.model.xL,s.idx.xL),
                   view(x,s.idx.xU),view(s.model.xU,s.idx.xU),
                   view(x,s.idx.xLs),view(s.model.xL,s.idx.xLs),
                   view(x,s.idx.xUs),view(s.model.xU,s.idx.xUs),
                   s.μ,s.opts.single_bnds_damping ? s.opts.κd : 0.,
                   view(x,s.idx.r),s.λ,s.ρ)
end

"""
    accept_step!(s::Solver)

Accept the current step, copying the candidate primals and duals into the current iterate.
"""
function accept_step!(s::Solver)
    s.x .= s.x⁺
    s.y .+= s.α*s.dy
    s.zL .+= s.αz*s.dzL
    s.zU .+= s.αz*s.dzU
    return nothing
end

"""
    small_search_direction(s::Solver)

Check if the current step is small (Sec. 3.9).
"""
function small_search_direction(s::Solver)
    return (maximum(abs.(s.dx)./(1.0 .+ abs.(s.x))) < 10.0*s.opts.ϵ_mach)
end

function relax_bounds!(s::Solver)
    for i in s.idx.xLs
        if s.x[i] - s.model.xL[i] < s.opts.ϵ_mach*s.μ
            s.model.xL[i] -= (s.opts.ϵ_mach^0.75)*max(1.0,s.model.xL[i])
            @warn "lower bound needs to be relaxed"
        end
    end

    for i in s.idx.xUs
        if s.model.xU[i] - s.x[i] < s.opts.ϵ_mach*s.μ
            s.model.xU[i] += (s.opts.ϵ_mach^0.75)*max(1.0,s.model.xU[i])
            @warn "upper bound needs to be relaxed"
        end
    end
end

"""
    NCSolver{T}

Complete interior point solver as described by the Ipopt paper.

# Fields
- `s`: interior point solver for the original problem
- `s`: interior point solver for the restoration phase
"""
struct NCSolver{T}
    s::Solver{T}
    s̄::Solver{T}
end

function NCSolver(x0,model;opts=Options{Float64}()) where T
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
    s = Solver(_x0,model_s,model,opts=opts)
    s̄ = RestorationSolver(s)

    NCSolver(s,s̄)
end

function update_quasi_newton!(s::Solver)
    if s.opts.quasi_newton != :none
        s.qn.∇L_prev .= s.qn.∇f_prev + s.qn.∇c_prev'*s.y
        s.qn.∇L_prev[s.idx.xL] -= s.zL
        s.qn.∇L_prev[s.idx.xU] += s.zU

        # s.model.mA > 0 && (s.qn.∇L_prev[s.idx.r] += s.λ + s.ρ*view(s.qn.x_prev,s.idx.r))
        #
        # # damping
        # if s.opts.single_bnds_damping
        #     κd = s.opts.κd
        #     μ = s.μ
        #     s.qn.∇L_prev[s.idx.xLs] .+= κd*μ
        #     s.qn.∇L_prev[s.idx.xUs] .-= κd*μ
        # end

        s.qn.∇f_prev .= get_∇f(s.model)
        s.qn.∇c_prev .= get_∇c(s.model)

        s.qn.∇L .= s.qn.∇f_prev + s.qn.∇c_prev'*s.y
        s.qn.∇L[s.idx.xL] -= s.zL
        s.qn.∇L[s.idx.xU] += s.zU

        # s.model.mA > 0 && (s.qn.∇L[s.idx.r] += s.λ + s.ρ*view(s.x,s.idx.r))
        #
        # # damping
        # if s.opts.single_bnds_damping
        #     κd = s.opts.κd
        #     μ = s.μ
        #     s.qn.∇L[s.idx.xLs] .+= κd*μ
        #     s.qn.∇L[s.idx.xUs] .-= κd*μ
        # end

        update_quasi_newton!(s.qn,s.x)
        s.∇²L .= copy(get_B(s.qn))
        s.model.mA > 0 && (view(s.∇²L,CartesianIndex.(s.idx.r,s.idx.r)) .+= s.ρ)
    end
    return nothing
end

function get_f(s::Solver,x)
    (s.opts.nlp_scaling ? s.df*get_f(s.model,x) : get_f(s.model,x))
end

function get_solution(s::NCSolver)
    return s.s.x[1:s.s.model_opt.n]
end
