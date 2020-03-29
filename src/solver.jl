mutable struct Solver{T}
    x::Vector{T}
    x⁺::Vector{T}

    xL::Vector{T}
    xU::Vector{T}
    xL_bool::Vector{Bool}
    xU_bool::Vector{Bool}

    x_soc::Vector{T}

    λ::Vector{T}
    zL::Vector{T}
    zU::Vector{T}

    n::Int
    nL::Int
    nU::Int
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

    restoration::Bool
    DR::SparseMatrixCSC{T,Int}

    opts::Options{T}
end

function Solver(x0,n,m,xL,xU,f_func,c_func,∇f_func,∇c_func; opts=opts{Float64}())

    # initialize primals
    x = zeros(n)

    x⁺ = zeros(n)

    # primal bounds
    xL_bool = ones(Bool,n)
    xU_bool = ones(Bool,n)

    for i = 1:n
        # relax bounds
        xL[i] = relax_bnd(xL[i],opts.ϵ_tol,:L)
        xU[i] = relax_bnd(xU[i],opts.ϵ_tol,:U)

        # boolean bounds
        if xL[i] < -1.0*opts.bnd_tol
            xL_bool[i] = 0
        end
        if xU[i] > opts.bnd_tol
            xU_bool[i] = 0
        end
    end

    for i = 1:n
        x[i] = init_x0(x0[i],xL[i],xU[i],opts.κ1,opts.κ2)
    end

    x_soc = zeros(n)

    nL = convert(Int,sum(xL_bool))
    nU = convert(Int,sum(xU_bool))

    zL = opts.zL0*ones(nL)
    zU = opts.zU0*ones(nU)

    H = spzeros(n+m,n+m)
    h = zeros(n+m)

    Hu = spzeros(n+m,n+m)
    hu = zeros(n+m)


    W = spzeros(n,n)
    ΣL = spzeros(n,n)
    ΣU = spzeros(n,n)
    A = spzeros(m,n)

    f = 0.
    ∇f = zeros(n)
    φ = 0.
    ∇φ = zeros(n)
    ∇L = zeros(n)
    c = zeros(m)
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

    θ = norm(c_func(x),1)
    θ_min = init_θ_min(θ)
    θ_max = init_θ_max(θ)

    θ_soc = 0.

    λ = opts.λ_init_ls ? init_λ(zL,zU,∇f_func(x),∇c_func(x),n,m,xL_bool,xU_bool,opts.λ_max) : zeros(m)

    sd = init_sd(λ,[zL;zU],n,m,opts.s_max)
    sc = init_sc([zL;zU],n,opts.s_max)

    filter = Tuple[]

    j = 0
    k = 0
    l = 0
    p = 0
    t = 0

    restoration = false
    DR = spzeros(0,0)

    Solver(x,x⁺,xL,xU,xL_bool,xU_bool,x_soc,λ,zL,zU,n,nL,nU,m,f_func,∇f_func,
        c_func,∇c_func,H,h,Hu,hu,W,ΣL,ΣU,A,f,∇f,φ,∇φ,∇L,c,c_soc,d,d_soc,dx,dλ,
        dzL,dzU,μ,α,αz,α_max,α_min,α_soc,β,τ,δw,δw_last,δc,θ,θ_min,θ_max,
        θ_soc,sd,sc,filter,j,k,l,p,t,restoration,DR,opts)
end

function eval_Eμ(x,λ,zL,zU,xL,xU,xL_bool,xU_bool,c,∇L,μ,sd,sc)
    return max(norm(∇L,Inf)/sd,
               norm(c,Inf),
               norm((x-xL)[xL_bool].*zL .- μ,Inf)/sc,
               norm((xU-x)[xU_bool].*zU .- μ,Inf)/sc)
end

eval_Eμ(μ,s::Solver) = eval_Eμ(s.x,s.λ,s.zL,s.zU,s.xL,s.xU,s.xL_bool,s.xU_bool,s.c,s.∇L,μ,s.sd,s.sc)

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
    s.∇L[s.xL_bool] .-= s.zL
    s.∇L[s.xU_bool] .+= s.zU
    return nothing
end

function eval_barrier!(s::Solver)
    s.φ = s.f
    s.φ -= s.μ*sum(log.((s.x - s.xL)[s.xL_bool]))
    s.φ -= s.μ*sum(log.((s.xU - s.x)[s.xU_bool]))

    s.∇φ .= s.∇f
    s.∇φ[s.xL_bool] .-= s.μ./(s.x - s.xL)[s.xL_bool]
    s.∇φ[s.xU_bool] .+= s.μ./(s.xU - s.x)[s.xU_bool]
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
function fraction_to_boundary_bnds(x,xL,xU,xL_bool,xU_bool,d,α,τ)
    return all((xU-(x + α*d))[xU_bool] .>= (1 - τ)*(xU-x)[xU_bool]) && all(((x + α*d)-xL)[xL_bool] .>= (1 - τ)*(x-xL)[xL_bool])
end

reset_z(z,x,μ,κΣ) = max(min(z,κΣ*μ/x),μ/(κΣ*x))

function reset_z!(s::Solver)
    for i = 1:s.nL
        s.zL[i] = reset_z(s.zL[i],s.x[s.xL_bool][i],s.μ,s.opts.κΣ)
    end

    for i = 1:s.nU
        s.zU[i] = reset_z(s.zU[i],s.x[s.xU_bool][i],s.μ,s.opts.κΣ)
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

function init_λ(zL,zU,∇f,∇c,n,m,xL_bool,xU_bool,λ_max)

    if m > 0
        H = [Matrix(I,n,n) ∇c';∇c zeros(m,m)]
        h = zeros(n+m)
        h[1:n] .= ∇f
        h[1:n][xL_bool] .-= zL
        h[1:n][xU_bool] .+= zU

        d = -H\h

        λ = d[n .+ (1:m)]

        if norm(λ,Inf) > λ_max || any(isnan.(λ))
            @warn "least-squares λ init failure:\n λ_max = $(norm(λ,Inf))"
            return zeros(m)
        else
            return λ
        end
    else
        return zeros(m)
    end
end

θ(x,s::Solver) = norm(s.c_func(x),1)

function barrier(x,xL,xU,xL_bool,xU_bool,μ,f_func)
    return (f_func(x) - μ*sum(log.((xU - x)[xU_bool])) - μ*sum(log.((x - xL)[xL_bool])))
end
barrier(x,s::Solver) = barrier(x,s.xL,s.xU,s.xL_bool,s.xU_bool,s.μ,s.f_func)

function update!(s::Solver)
    s.x .= s.x⁺
    s.λ .= s.λ + s.α*s.dλ
    s.zL .= s.zL + s.αz*s.dzL
    s.zU .= s.zU + s.αz*s.dzU

    reset_z!(s)
    return nothing
end

function relax_bnd(x_bnd,ϵ_tol,bnd_type)
    if bnd_type == :L
        return x_bnd - ϵ_tol*max(1.0,abs(x_bnd))
    elseif bnd_type == :U
        return x_bnd + ϵ_tol*max(1.0,abs(x_bnd))
    else
        error("bound type error")
    end
end

# function eval_Fμ!(Fμ,x,λ,zL,zU,xL,xU,xL_bool,xU_bool,c,∇L,μ,n,m,nL,nU)
#     Fμ[1:n] = ∇L
#     Fμ[n .+ (1:m)] = c
#     Fμ[(n+m) .+ (1:nL)] = (x-xL)[xL_bool].*zL .- μ
#     Fμ[(n+m+nL) .+ (1:nU)] = (xU-x)[xU_bool].*zU .- μ
#     return nothing
# end
