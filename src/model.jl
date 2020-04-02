# min  f(x)
# s.t. xL < x < xU
#      c(x) = 0
#

struct Model{T}
    # primal dimension
    n::Int

    # constraints
    m::Int

    # primal bounds
    nL::Int
    nU::Int

    xL::Vector{T}
    xU::Vector{T}
    xL_bool::Vector{Bool}
    xU_bool::Vector{Bool}
    xLs_bool::Vector{Bool}
    xUs_bool::Vector{Bool}

    # objective
    f::Function
    ∇f::Function

    # constraints
    c::Function
    ∇c::Function

    # Hessian of Lagrangian
    ∇²L::Function
    ∇²L_autodif::Bool
end

function Model(n,m,xL,xU,f,c,∇f,∇c; bnd_tol=1.0e8)
    # primal bounds
    xL_bool = ones(Bool,n)
    xU_bool = ones(Bool,n)
    xLs_bool = zeros(Bool,n)
    xUs_bool = zeros(Bool,n)

    for i = 1:n

        # boolean bounds
        if xL[i] < -1.0*bnd_tol
            xL_bool[i] = 0
        end

        if xU[i] > bnd_tol
            xU_bool[i] = 0
        end

        # single bounds
        if xL_bool[i] == 1 && xU_bool[i] == 0
            xLs_bool[i] = 1
        else
            xLs_bool[i] = 0
        end

        if xL_bool[i] == 0 && xU_bool[i] == 1
            xUs_bool[i] = 1
        else
            xUs_bool[i] = 0
        end
    end

    nL = convert(Int,sum(xL_bool))
    nU = convert(Int,sum(xU_bool))

    ∇²L(x,λ) = nothing

    Model(n,m,nL,nU,xL,xU,xL_bool,xU_bool,xLs_bool,xUs_bool,f,∇f,c,∇c,∇²L,true)
end


mutable struct SolverModel{T}
    model::Model{T}

    x::Vector{T}
    x⁺::Vector{T}
    x_soc::Vector{T}

    λ::Vector{T}
    zL::Vector{T}
    zU::Vector{T}

    H::SparseMatrixCSC{T,Int}
    h::Vector{T}

    Hu::SparseMatrixCSC{T,Int}
    hu::Vector{T}

    W::SparseMatrixCSC{T,Int}
    ΣL::SparseMatrixCSC{T,Int}
    ΣU::SparseMatrixCSC{T,Int}
    A::SparseMatrixCSC{T,Int}

    f::T
    ∇f::Vector{T}
    φ::T
    ∇φ::Vector{T}
    ∇L::Vector{T}
    c::Vector{T}
    c_soc::Vector{T}

    d::Vector{T}
    d_soc::Vector{T}

    dx
    dλ
    dzL
    dzU

    μ::T

    α::T
    αz::T
    α_max::T
    α_min::T
    α_soc::T
    β::T

    τ::T

    δw::T
    δw_last::T
    δc::T

    θ::T
    θ_min::T
    θ_max::T
    θ_soc::T

    sd::T
    sc::T

    filter::Vector{Tuple}

    j::Int
    k::Int
    l::Int
    p::Int
    t::Int
    small_search_direction_cnt::Int

    restoration::Bool
    DR::SparseMatrixCSC{T,Int}

    x_copy::Vector{T}
    λ_copy::Vector{T}
    zL_copy::Vector{T}
    zU_copy::Vector{T}

    Fμ::Vector{T}

    Dx::SparseMatrixCSC{T,Int}
    df::T
    Dc::SparseMatrixCSC{T,Int}

    opts::Options{T}
end

function SolverModel(x0::Vector{T},model::Model{T}; opts=Options{T}()) where T
    n = model.n
    m = model.m
    nL = model.nL
    nU = model.nU

    # initialize primals
    x = zeros(n)
    for i = 1:n
        x[i] = init_x0(x0[i],model.xL[i],model.xU[i],opts.κ1,opts.κ2)
    end

    x⁺ = zeros(n)
    x_soc = zeros(n)

    zL = opts.zL0*ones(nL)
    zU = opts.zU0*ones(nU)

    H = spzeros(n+m,n+m)
    h = zeros(n+m)

    Hu = spzeros(n+m+nL+nU,n+m+nL+nU)
    hu = zeros(n+m+nL+nU)

    W = spzeros(n,n)
    ΣL = spzeros(n,n)
    ΣU = spzeros(n,n)
    A = sparse(∇c_func(x))

    f = model.f(x)
    ∇f = model.∇f(x)
    φ = 0.
    ∇φ = zeros(n)
    ∇L = zeros(n)
    c = model.c(x)
    c_soc = zeros(m)

    d = zeros(n+m+nL+nU)
    d_soc = zeros(n+m+nL+nU)

    dx = view(d,1:n)
    dλ = view(d,n .+ (1:m))
    dzL = view(d,n+m .+ (1:nL))
    dzU = view(d,n+m+nL .+ (1:nU))

    μ = copy(opts.μ0)

    α = 1.0
    αz = 1.0
    α_max = 1.0
    α_min = 1.0
    α_soc = 1.0
    β = 1.0

    τ = update_τ(μ,opts.τ_min)

    δw = 0.
    δw_last = 0.
    δc = 0.

    θ = norm(c,1)
    θ_min = init_θ_min(θ)
    θ_max = init_θ_max(θ)

    θ_soc = 0.

    λ = opts.λ_init_ls ? init_λ(zL,zU,∇f,A,n,m,model.xL_bool,model.xU_bool,opts.λ_max) : zeros(m)

    h = zeros(n+m)

    sd = init_sd(λ,[zL;zU],n,m,opts.s_max)
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

    x_copy = zero(x)
    λ_copy = zero(λ)
    zL_copy = zero(zL)
    zU_copy = zero(zU)

    Fμ = zeros(n+m+nL+nU)

    Dx = init_Dx(n)
    df = init_df(∇f_func(x),opts.g_max)
    Dc = init_Dc(m,∇c_func(x),opts.g_max)

    SolverModel(model,x,x⁺,x_soc,λ,zL,zU,H,h,Hu,hu,W,ΣL,ΣU,A,f,∇f,φ,∇φ,∇L,c,c_soc,d,d_soc,
        dx,dλ,dzL,dzU,μ,α,αz,α_max,α_min,α_soc,β,τ,δw,δw_last,δc,θ,θ_min,θ_max,
        θ_soc,sd,sc,filter,j,k,l,p,t,small_search_direction_cnt,restoration,DR,
        x_copy,λ_copy,zL_copy,zU_copy,Fμ,Dx,df,Dc,
        opts)
end

function RestorationSolverModel(s::SolverModel)
    s.c .= s.model.c(s.x)

    opts = copy(s.opts)
    opts.λ_init_ls = false
    opts.μ0 = max(s.μ,norm(s.c,Inf))

    n̄ = s.model.n + 2s.model.m
    m̄ = s.model.m

    x̄L = zeros(n̄)
    x̄L[1:s.model.n] = s.model.xL

    x̄U = Inf*ones(n̄)
    x̄U[1:s.model.n] = s.model.xU

    ζ = sqrt(opts.μ0)
    DR = init_DR(s.x,s.model.n)
    xR = copy(s.x)

    f_func(x) = s.opts.ρ*sum(x[s.model.n .+ (1:2s.model.m)]) + 0.5*ζ*norm(DR*(x[1:s.model.n] - xR))^2
    ∇f_func(x) = ForwardDiff.gradient(f_func,x)

    c_func(x) = s.model.c(x[1:s.model.n]) - x[s.model.n .+ (1:s.model.m)] + x[(s.model.n+s.model.m) .+ (1:s.model.m)]
    ∇c_func(x) = [s.model.∇c(x[1:s.model.n]) -I I]

    model = Model(n̄,m̄,x̄L,x̄U,f_func,c_func,∇f_func,∇c_func,bnd_tol=s.opts.bnd_tol)

    x̄ = zeros(n̄)
    x̄[1:s.model.n] = copy(s.x)

    # initialize p,n
    for i = 1:s.model.m
        x̄[s.model.n + s.model.m + i] = init_n(s.c[i],opts.μ0,s.opts.ρ)
    end

    for i = 1:s.model.m
        x̄[s.model.n + i] = init_p(x̄[s.model.n + s.model.m + i],s.c[i])
    end
    #
    s̄ = SolverModel(x̄,model,opts=opts)

    # initialize zL, zU, zp, zn
    for i = 1:s.model.nL
        s̄.zL[i] = min(opts.ρ,s.zL[i])
    end

    for i = 1:s.model.nU
        s̄.zU[i] = min(opts.ρ,s.zU[i])
    end

    s̄.zL[s.model.nL .+ (1:2s.model.m)] = opts.μ0./s̄.x[s.model.n .+ (1:2s.model.m)]

    s̄.DR = DR

    s̄.restoration = true

    return s̄
end

struct InteriorPointSolver{T}
    s1::SolverModel{T}
    s2::SolverModel{T}
end

function InteriorPointSolver(x0::Vector{T},model::Model{T}; opts=Options{T}()) where T
    s1 = SolverModel(x0,model,opts=opts)
    s2 = RestorationSolverModel(s1)

    InteriorPointSolver(s1,s2)
end
