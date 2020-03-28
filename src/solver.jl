mutable struct Solver{T}
    x::Vector{T}
    x⁺::Vector{T}
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

    Hu::SparseMatrixCSC{T,Int}
    hu::Vector{T}

    W::SparseMatrixCSC{T,Int}
    Σl::SparseMatrixCSC{T,Int}
    Σu::SparseMatrixCSC{T,Int}
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
    dzl
    dzu

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

    restoration::Bool
    DR::SparseMatrixCSC{T,Int}

    opts::Options{T}
end

function Solver(x0,n,m,xl,xu,f_func,c_func,∇f_func,∇c_func; opts=opts{Float64}())

    # initialize primals
    x = zeros(n)
    for i = 1:n
        x[i] = init_x0(x0[i],xl[i],xu[i],opts.κ1,opts.κ2)
    end

    x⁺ = zeros(n)

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

    H = spzeros(n+m,n+m)
    h = zeros(n+m)

    Hu = spzeros(n+m,n+m)
    hu = zeros(n+m)


    W = spzeros(n,n)
    Σl = spzeros(n,n)
    Σu = spzeros(n,n)
    A = spzeros(m,n)

    f = 0.
    ∇f = zeros(n)
    φ = 0.
    ∇φ = zeros(n)
    ∇L = zeros(n)
    c = zeros(m)
    c_soc = zeros(m)

    d = zeros(n+m+nl+nu)
    d_soc = zeros(n+m+nl+nu)

    dx = view(d,1:n)
    dλ = view(d,n .+ (1:m))
    dzl = view(d,n+m .+ (1:nl))
    dzu = view(d,n+m+nl .+ (1:nu))

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
    δw_last = 0.
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

    restoration = false
    DR = spzeros(0,0)

    Solver(x,x⁺,xl,xu,xl_bool,xu_bool,x_soc,λ,zl,zu,n,nl,nu,m,f_func,∇f_func,
        c_func,∇c_func,H,h,Hu,hu,W,Σl,Σu,A,f,∇f,φ,∇φ,∇L,c,c_soc,d,d_soc,dx,dλ,
        dzl,dzu,μ,α,αz,α_max,α_min,α_soc,β,update,τ,δw,δw_last,δc,θ,θ_min,θ_max,
        θ_soc,sd,sc,filter,j,k,l,p,t,restoration,DR,opts)
end

function eval_Eμ(x,λ,zl,zu,xl,xu,xl_bool,xu_bool,c,∇L,μ,sd,sc)
    return max(norm(∇L,Inf)/sd,
               norm(c,Inf),
               norm((x-xl)[xl_bool].*zl .- μ,Inf)/sc,
               norm((xu-x)[xu_bool].*zu .- μ,Inf)/sc)
end

eval_Eμ(μ,s::Solver) = eval_Eμ(s.x,s.λ,s.zl,s.zu,s.xl,s.xu,s.xl_bool,s.xu_bool,s.c,s.∇L,μ,s.sd,s.sc)

function eval_objective!(s::Solver)
    s.f = s.f_func(s.x)
    s.∇f .= s.∇f_func(s.x)
    return nothing
end

function eval_constraints!(s::Solver)
    s.c .= s.c_func(s.x)
    s.A .= s.∇c_func(s.x)

    s.θ = norm(s.c,1)
    return nothing
end

function eval_lagrangian!(s::Solver)
    s.∇L .= s.∇f
    s.∇L .+= s.A'*s.λ
    s.∇L[s.xl_bool] .-= s.zl
    s.∇L[s.xu_bool] .+= s.zu
    return nothing
end

function eval_barrier!(s::Solver)
    s.φ = s.f
    s.φ -= s.μ*sum(log.((s.x - s.xl)[s.xl_bool]))
    s.φ -= s.μ*sum(log.((s.xu - s.x)[s.xu_bool]))

    s.∇φ .= s.∇f
    s.∇φ[s.xl_bool] .-= s.μ./(s.x - s.xl)[s.xl_bool]
    s.∇φ[s.xu_bool] .+= s.μ./(s.xu - s.x)[s.xu_bool]
    return nothing
end

function eval_iterate!(s::Solver)
    eval_objective!(s)
    eval_constraints!(s)
    eval_lagrangian!(s)
    eval_barrier!(s)
    return nothing
end

function init_sd(λ,z,n,m,s_max)
    sd = max(s_max,(norm(λ,1) + norm(z,1))/(n+m))/s_max
    return sd
end

function init_sc(z,n,s_max)
    sc = max(s_max,norm(z,1)/n)/s_max
    return sc
end

update_μ(μ,κμ,θμ,ϵ_tol) = max(ϵ_tol/10.,min(κμ*μ,μ^θμ))
function update_μ!(s::Solver)
    s.μ = update_μ(s.μ,s.opts.κμ,s.opts.θμ,s.opts.ϵ_tol)
    return nothing
end

update_τ(μ,τ_min) = max(τ_min,1.0-μ)
function update_τ!(s::Solver)
    s.τ = update_τ(s.μ,s.opts.τ_min)
    return nothing
end

fraction_to_boundary(x,d,α,τ) = all(x + α*d .>= (1 - τ)*x)
function fraction_to_boundary_bnds(x,xl,xu,xl_bool,xu_bool,d,α,τ)
    return all((xu-(x + α*d))[xu_bool] .>= (1 - τ)*(xu-x)[xu_bool]) && all(((x + α*d)-xl)[xl_bool] .>= (1 - τ)*(x-xl)[xl_bool])
end

reset_z(z,x,μ,κΣ) = max(min(z,κΣ*μ/x),μ/(κΣ*x))

function reset_z!(s::Solver)
    for i = 1:s.nl
        s.zl[i] = reset_z(s.zl[i],s.x[s.xl_bool][i],s.μ,s.opts.κΣ)
    end

    for i = 1:s.nu
        s.zu[i] = reset_z(s.zu[i],s.x[s.xu_bool][i],s.μ,s.opts.κΣ)
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

function init_x0(x,xl,xu,κ1,κ2)
    pl = min(κ1*max(1.0,abs(xl)),κ2*(xu-xl))
    pu = min(κ1*max(1.0,abs(xu)),κ2*(xu-xl))

    # projection
    if x < xl+pl
        x = xl+pl
    elseif x > xu-pu
        x = xu-pu
    end
    return x
end

function init_λ(zl,zu,∇f,∇c,n,m,xl_bool,xu_bool,λ_max)

    H = [Matrix(I,n,n) ∇c';∇c zeros(m,m)]
    h = zeros(n+m)
    h[1:n] .= ∇f
    h[1:n][xl_bool] .-= zl
    h[1:n][xu_bool] .+= zu

    d = -H\h

    λ = d[n .+ (1:m)]

    if norm(λ,Inf) > λ_max || any(isnan.(λ))
        @warn "least-squares λ init failure:\n λ_max = $(norm(λ,Inf))"
        return zeros(m)
    else
        return λ
    end
end

θ(x,s::Solver) = norm(s.c_func(x),1)

function barrier(x,xl,xu,xl_bool,xu_bool,μ,f_func)
    return (f_func(x) - μ*sum(log.((xu - x)[xu_bool])) - μ*sum(log.((x - xl)[xl_bool])))
end
barrier(x,s::Solver) = barrier(x,s.xl,s.xu,s.xl_bool,s.xu_bool,s.μ,s.f_func)

function update!(s::Solver)
    if s.update == :nominal
        s.x .= s.x + s.α*s.dx
    elseif s.update == :soc
        s.x .= s.x + s.α_soc*s.d_soc[1:s.n]
        s.update = :nominal # reset update
    else
        error("update error")
    end
    s.λ .= s.λ + s.α*s.dλ
    s.zl .= s.zl + s.αz*s.dzl
    s.zu .= s.zu + s.αz*s.dzu

    reset_z!(s)
    return nothing
end

function eval_Fμ!(Fμ,x,λ,zl,zu,xl,xu,xl_bool,xu_bool,c,∇L,μ,n,m,nl,nu)
    Fμ[1:n] = ∇L
    Fμ[n .+ (1:m)] = c
    Fμ[(n+m) .+ (1:nl)] = (x-xl)[xl_bool].*zl .- μ
    Fμ[(n+m+nl) .+ (1:nu)] = (xu-x)[xu_bool].*zu .- μ
    return nothing
end
