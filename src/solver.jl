mutable struct Solver{T}
    x::Vector{T}
    xl::Vector{T}
    xu::Vector{T}
    xl_bool::Vector{Bool}
    xu_bool::Vector{Bool}
    x_soc::Vector{T}

    λ::Vector{T}
    zl::Vector{T}
    zu::Vector{T}

    n::Int
    nl::Int
    nu::Int
    m::Int

    f_func::Function
    ∇f_func::Function
    c_func::Function
    ∇c_func::Function

    H::SparseMatrixCSC{T,Int}
    h::Vector{T}

    W::SparseMatrixCSC{T,Int}
    Σl::SparseMatrixCSC{T,Int}
    Σu::SparseMatrixCSC{T,Int}
    A::SparseMatrixCSC{T,Int}

    ∇f::Vector{T}
    ∇φ::Vector{T}
    ∇L::Vector{T}
    c::Vector{T}
    c_soc::Vector{T}

    d::Vector{T}
    d_soc::Vector{T}

    dzl::Vector{T}
    dzu::Vector{T}

    μ::T
    α::T
    αz::T
    α_max::T
    α_min::T
    α_soc::T
    β::T

    update::Symbol

    τ::T

    δw::T
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

    opts::Options{T}
end

function Solver(x0,n,m,xl,xu,f_func,c_func,∇f_func,∇c_func; opts=opts{Float64}())

    # initialize primals
    x = zeros(n)
    for i = 1:n
        x[i] = init_x0(x0[i],xl[i],xu[i],opts.κ1,opts.κ2)
    end

    # check primal bounds
    xl_bool = ones(Bool,n)
    xu_bool = ones(Bool,n)

    for i = 1:n
        if xl[i] < -1.0*opts.bnd_tol
            xl_bool[i] = 0
        end
        if xu[i] > opts.bnd_tol
            xu_bool[i] = 0
        end
    end

    x_soc = zeros(n)

    nl = convert(Int,sum(xl_bool))
    nu = convert(Int,sum(xu_bool))

    zl = opts.zl0*ones(nl)
    zu = opts.zu0*ones(nu)

    # ∇f_func(x) = ForwardDiff.gradient(f_func,x)
    # ∇c_func(x) = m > 1 ? ForwardDiff.jacobian(c_func,x) : ForwardDiff.gradient(c_func,x)

    H = spzeros(n+m,n+m)
    h = zeros(n+m)

    W = spzeros(n,n)
    Σl = spzeros(n,n)
    Σu = spzeros(n,n)
    A = spzeros(m,n)

    ∇f = zeros(n)
    ∇φ = zeros(n)
    ∇L = zeros(n)
    c = zeros(m)
    c_soc = zeros(m)

    d = zeros(n+m)
    d_soc = zeros(n+m)

    dzl = zeros(nl)
    dzu = zeros(nu)

    μ = copy(opts.μ0)

    α = 1.0
    αz = 1.0
    α_max = 1.0
    α_min = 1.0
    α_soc = 1.0
    β = 1.0

    update = :nominal

    τ = update_τ(μ,opts.τ_min)

    δw = 0.
    δc = 0.

    θ = norm(c_func(x),1)
    θ_min = init_θ_min(θ)
    θ_max = init_θ_max(θ)

    θ_soc = 0.

    λ = opts.λ_init_ls ? init_λ(zl,zu,∇f_func(x),∇c_func(x),n,m,xl_bool,xu_bool,opts.λ_max) : zeros(m)

    sd = init_sd(λ,[zl;zu],n,m,opts.s_max)
    sc = init_sc([zl;zu],n,opts.s_max)

    filter = Tuple[]

    j = 0
    k = 0
    l = 0
    p = 0
    t = 0


    Solver(x,xl,xu,xl_bool,xu_bool,x_soc,λ,zl,zu,n,nl,nu,m,f_func,∇f_func,c_func,
        ∇c_func,H,h,W,Σl,Σu,A,∇f,∇φ,∇L,c,c_soc,d,d_soc,dzl,dzu,μ,α,αz,α_max,α_min,
        α_soc,β,update,τ,δw,δc,θ,θ_min,θ_max,θ_soc,sd,sc,filter,j,k,l,p,t,opts)
end

function eval_Eμ(μ,s::Solver)
    s.c .= s.c_func(s.x)
    s.∇f .= s.∇f_func(s.x)
    s.A .= s.∇c_func(s.x)
    s.∇L .= s.∇f + s.A'*s.λ
    s.∇L[s.xl_bool] .-= s.zl
    s.∇L[s.xu_bool] .+= s.zu
    return eval_Eμ(s.x,s.λ,s.zl,s.zu,s.xl,s.xu,s.xl_bool,s.xu_bool,s.c,s.∇L,μ,
        s.sd,s.sc)
end
