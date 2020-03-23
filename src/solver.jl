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

    xR::Vector{T}
    p_res::Vector{T}
    n_res::Vector{T}
    zp_res::Vector{T}
    zn_res::Vector{T}
    dp::Vector{T}
    dn::Vector{T}
    dzp::Vector{T}
    dzn::Vector{T}
    Σp::SparseMatrixCSC{T,Int}
    Σn::SparseMatrixCSC{T,Int}
    μ_res::T
    ζ::T
    DR::SparseMatrixCSC{T,Int}

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

    # feasibility restoration
    xR = zeros(n)
    p_res = ones(m)
    n_res = ones(m)
    zp_res = ones(m)
    zn_res = ones(m)
    dp = ones(m)
    dn = ones(m)
    dzp = ones(m)
    dzn = ones(m)
    Σp = spzeros(m,m)
    Σn = spzeros(m,m)
    μ_res = copy(opts.μ0)
    ζ = 1.0
    DR = spzeros(n,n)

    Solver(x,xl,xu,xl_bool,xu_bool,x_soc,λ,zl,zu,n,nl,nu,m,f_func,∇f_func,c_func,
        ∇c_func,H,h,W,Σl,Σu,A,∇f,∇φ,∇L,c,c_soc,d,d_soc,dzl,dzu,μ,α,αz,α_max,α_min,
        α_soc,β,update,τ,δw,δc,θ,θ_min,θ_max,θ_soc,sd,sc,filter,j,k,l,p,t,
        xR,p_res,n_res,zp_res,zn_res,dp,dn,dzp,dzn,Σp,Σn,μ_res,ζ,DR,opts)
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

function eval_Eμ_restoration(μ,s::Solver)
    s.c .= s.c_func(s.x) - s.p_res + s.n_res
    s.A .= s.∇c_func(s.x)
    s.∇L .= s.ζ*s.DR'*s.DR*(s.x - s.xR) + s.A'*s.λ
    s.∇L[s.xl_bool] .-= s.zl
    s.∇L[s.xu_bool] .+= s.zu
    return eval_Eμ_restoration(s.x,s.p_res,s.n_res,s.λ,s.zl,s.zu,s.zp_res,s.zn_res,s.xl,s.xu,s.xl_bool,
        s.xu_bool,s.c,s.∇L,μ,s.opts.ρ,s.sd,s.sc)
end

function eval_Fμ(x,λ,zl,zu,s::Solver)
    s.c .= s.c_func(x)
    s.∇f .= s.∇f_func(x)
    s.A .= s.∇c_func(x)
    s.∇L .= s.∇f + s.A'*λ
    s.∇L[s.xl_bool] .-= zl
    s.∇L[s.xu_bool] .+= zu
    return eval_Fμ(x,λ,zl,zu,s.xl,s.xu,s.xl_bool,s.xu_bool,s.c,s.∇L,s.μ)
end

function search_direction!(s::Solver)
    ∇L(x) = s.∇f_func(x) + s.∇c_func(x)'*s.λ
    s.W .= ForwardDiff.jacobian(∇L,s.x)
    s.Σl[CartesianIndex.((1:s.n)[s.xl_bool],(1:s.n)[s.xl_bool])] .= s.zl./((s.x - s.xl)[s.xl_bool])
    s.Σu[CartesianIndex.((1:s.n)[s.xu_bool],(1:s.n)[s.xu_bool])] .= s.zu./((s.xu - s.x)[s.xu_bool])

    s.c .= s.c_func(s.x)
    s.A .= s.∇c_func(s.x)

    s.∇φ .= s.∇f_func(s.x)
    s.∇φ[s.xl_bool] .-= s.μ./(s.x - s.xl)[s.xl_bool]
    s.∇φ[s.xu_bool] .+= s.μ./(s.xu - s.x)[s.xu_bool]

    s.H[1:s.n,1:s.n] .= (s.W + s.Σl + s.Σu + s.δw*I)
    s.H[1:s.n,s.n .+ (1:s.m)] .= s.A'
    s.H[s.n .+ (1:s.m),1:s.n] .= s.A
    s.H[s.n .+ (1:s.m),s.n .+ (1:s.m)] .= -s.δc*Matrix(I,s.m,s.m)

    s.h[1:s.n] .= s.∇φ + s.A'*s.λ
    s.h[s.n .+ (1:s.m)] .= s.c

    s.d .= -s.H\s.h
    s.dzl .= -s.zl./((s.x - s.xl)[s.xl_bool]).*s.d[1:n][s.xl_bool] - s.zl + s.μ./((s.x - s.xl)[s.xl_bool])
    s.dzu .= s.zu./((s.xu - s.x)[s.xu_bool]).*s.d[1:n][s.xu_bool] - s.zu + s.μ./((s.xu - s.x)[s.xu_bool])
    return nothing
end

function α_min!(s::Solver)
    s.∇φ .= s.∇f_func(s.x)
    s.∇φ[s.xl_bool] .-= s.μ./(s.x - s.xl)[s.xl_bool]
    s.∇φ[s.xu_bool] .+= s.μ./(s.xu - s.x)[s.xu_bool]
    s.α_min = update_α_min(s.d[1:s.n],s.θ,s.∇φ,s.θ_min,s.opts.δ,s.opts.γα,s.opts.γθ,s.opts.γφ,s.opts.sθ,s.opts.sφ)

    # println("α_min: $(s.α_min)")
    return nothing
end

function barrier(x,s::Solver)
    return barrier(x,s.xl,s.xu,s.xl_bool,s.xu_bool,s.μ,s.f_func)
end

function θ(x,s::Solver)
    return norm(s.c_func(x),1)
end

function α_max!(s::Solver)
    α_min!(s)

    s.α_max = 1.0
    while !fraction_to_boundary_bnds(s.x,s.xl,s.xu,s.d[1:s.n],s.α_max,s.τ)
        s.α_max *= 0.5
        println("α = $(s.α_max)")
        if s.α_max < s.α_min
            error("α < α_min")
        end
    end
    s.α = copy(s.α_max)

    s.αz = 1.0
    while !fraction_to_boundary(s.zl,s.dzl,s.αz,s.τ)
        s.αz *= 0.5
        println("αzl = $(s.αz)")
        if s.αz < s.α_min
            error("αzl < α_min")
        end
    end

    while !fraction_to_boundary(s.zu,s.dzu,s.αz,s.τ)
        s.αz *= 0.5
        println("αzu = $(s.αz)")
        if s.αz < s.α_min
            error("αzu < α_min")
        end
    end

    return nothing
end

function α_max_restoration!(s::Solver)

    s.α_max = 1.0
    while !fraction_to_boundary_bnds(s.x,s.xl,s.xu,s.d[1:s.n],s.α_max,s.τ)
        s.α_max *= 0.5
        println("α = $(s.α_max)")
        if s.α_max < s.α_min
            error("α < α_min")
        end
    end

    # p
    while !fraction_to_boundary(s.p_res,s.dp,s.α_max,s.τ)
        s.α_max *= 0.5
        println("α = $(s.α_max)")
        if s.α_max < s.α_min
            error("α < α_min")
        end
    end

    # n
    while !fraction_to_boundary(s.n_res,s.dn,s.α_max,s.τ)
        s.α_max *= 0.5
        println("α = $(s.α_max)")
        if s.α_max < s.α_min
            error("α < α_min")
        end
    end
    s.α = copy(s.α_max)

    s.αz = 1.0
    while !fraction_to_boundary(s.zl,s.dzl,s.αz,s.τ)
        s.αz *= 0.5
        println("αzl = $(s.αz)")
        if s.αz < s.α_min
            error("αzl < α_min")
        end
    end

    while !fraction_to_boundary(s.zu,s.dzu,s.αz,s.τ)
        s.αz *= 0.5
        println("αzu = $(s.αz)")
        if s.αz < s.α_min
            error("αzu < α_min")
        end
    end

    while !fraction_to_boundary(s.zp_res,s.dzp,s.αz,s.τ)
        s.αz *= 0.5
        println("αzp = $(s.αz)")
        if s.αz < s.α_min
            error("αzp < α_min")
        end
    end

    while !fraction_to_boundary(s.zn_res,s.dzn,s.αz,s.τ)
        s.αz *= 0.5
        println("αzn = $(s.αz)")
        if s.αz < s.α_min
            error("αzn < α_min")
        end
    end

    return nothing
end

function β_max!(s::Solver)

    s.β = 1.0
    while !fraction_to_boundary_bnds(s.x,s.xl,s.xu,s.d[1:s.n],s.β,s.τ)
        s.β *= 0.5
        println("β = $(s.β)")
        if s.β < 1.0e-32
            error("β < 1e-32 ")
        end
    end

    while !fraction_to_boundary(s.zl,s.dzl,s.β,s.τ)
        s.β *= 0.5
        println("β = $(s.β)")
        if s.β < 1.0e-32
            error("β < 1e-32 ")
        end
    end

    while !fraction_to_boundary(s.zu,s.dzu,s.β,s.τ)
        s.β *= 0.5
        println("β = $(s.β)")
        if s.β < 1.0e-32
            error("β < 1e-32 ")
        end
    end

    return nothing
end

function switching_condition(s::Solver)
    s.∇φ .= s.∇f_func(s.x)
    s.∇φ[s.xl_bool] .-= s.μ./(s.x - s.xl)[s.xl_bool]
    s.∇φ[s.xu_bool] .+= s.μ./(s.xu - s.x)[s.xu_bool]
    s.θ = θ(s.x,s)
    return switching_condition(s.∇φ,s.d[1:s.n],s.α,s.opts.sφ,s.opts.δ,s.θ,s.opts.sθ)
end

function sufficient_progress(s::Solver)
    return sufficient_progress(θ(s.x + s.α*s.d[1:s.n],s),θ(s.x,s),
        barrier(s.x + s.α*s.d[1:s.n],s),barrier(s.x,s),s.opts.γθ,s.opts.γφ)
end

function armijo(x⁺,s::Solver)
    s.∇φ .= s.∇f_func(s.x)
    s.∇φ[s.xl_bool] .-= s.μ./(s.x - s.xl)[s.xl_bool]
    s.∇φ[s.xu_bool] .+= s.μ./(s.xu - s.x)[s.xu_bool]
    return armijo_condtion(barrier(x⁺,s),barrier(s.x,s),
        s.opts.ηφ,s.α,s.∇φ,s.d[1:s.n])
end

function check_filter(θ,φ,s::Solver)
    len = length(s.filter)
    cnt = 0
    println("θ: $θ")
    println("φ: $φ")
    for f in s.filter
        println("f(θ,φ): ($(f[1]),$(f[2]))")
        if θ < f[1] || φ < f[2]
        # if θ < (1.0 - s.opts.γθ)*f[1] || φ < f[2] - s.opts.γφ*f[1]
            cnt += 1
        end
    end
    println("cnt: $cnt")
    println("len: $len")
    if cnt == len
        return true
    else
        return false
    end
end

function add_to_filter!(p,s::Solver)
    f = s.filter
    if isempty(f)
        push!(f,p)
        return nothing
    end

    # check that new point is not dominated
    len = length(f)
    for _p in f
        if p[1] >= _p[1] && p[2] >= _p[2]
            len -= 1
        end
    end

    # remove filter's points dominated by new point
    if length(f) == len
        _f = copy(f)
        empty!(f)
        push!(f,p)
        for _p in _f
            if !(_p[1] >= p[1] && _p[2] >= p[2])
                push!(f,_p)
            end
        end
    end
    return nothing
end

function augment_filter!(s::Solver)
    if s.update == :nominal
        x⁺ = s.x + s.α*s.d[1:s.n]
    elseif s.update == :soc
        x⁺ = s.x + s.α_soc*s.d_soc[1:s.n]
    else
        error("update error in augment filter")
    end
    θ⁺ = θ(x⁺,s)
    φ⁺ = barrier(x⁺,s)

    if !switching_condition(s) || !armijo(x⁺,s)
        add_to_filter!((θ⁺,φ⁺),s)
    end

    return nothing
end

function second_order_correction(s::Solver)
    s.p = 1
    θ_soc = θ(s.x,s)
    c_soc = s.α_max*c_func(s.x) + c_func(s.x + s.α_max*s.d[1:s.n])
    s.h[s.n .+ (1:s.m)] .= c_soc
    s.d_soc = -s.H\s.h

    s.α_soc = 1.0
    while !fraction_to_boundary_bnds(s.x,s.xl,s.xu,s.d_soc[1:s.n],s.α_soc,s.τ)
        s.α_soc *= 0.5
        println("α = $(s.α_max)")
        if s.α_soc < s.α_min
            error("α < α_min")
        end
    end

    while true
        if check_filter(θ(s.x + s.α_soc*s.d_soc[1:s.n],s),barrier(s.x + s.α_soc*s.d_soc[1:s.n],s),s)
            if (θ(s.x,s) < s.θ_min && switching_condition(s))
                if armijo(s.x + s.α_soc*s.d_soc[1:s.n],s)
                    s.update = :soc
                    s.α = s.α_soc
                    break
                end
            # case 2
            else
                if sufficient_progress(s)
                    s.update = :soc
                    s.α = s.α_soc
                    break
                end
            end
        end

        if s.p == s.opts.p_max || θ(s.x + s.α_soc*s.d_soc[1:s.n],s) > s.opts.κ_soc*θ_soc
            @warn "second order correction failure"
            break
        end

        s.p += 1

        θ_soc = θ(s.x + s.α_soc*s.d_soc[1:s.n],s)
        s.c_soc = s.α_soc*s.c_soc + s.c_func(s.x + s.α_soc*s.d_soc[1:s.n])

        s.h[s.n .+ (1:s.m)] .= c_soc
        s.d_soc = -s.H\s.h

        s.α_soc = 1.0
        while !fraction_to_boundary_bnds(s.x,s.xl,s.xu,s.d_soc[1:s.n],s.α_soc,s.τ)
            s.α_soc *= 0.5
            println("α = $(s.α_max)")
            if s.α_soc < s.α_min
                error("α_soc < α_min")
            end
        end
    end

    if s.update == :soc
        println("second order correction: success")
        return true
    else
        println("second order correction: failure")
        return false
    end
end

function check_kkt_error(s::Solver)
    Fμ = norm(eval_Fμ(s.x,s.λ,s.zl,s.zu,s),1)
    Fμ⁺ = norm(eval_Fμ(s.x + s.β*s.d[1:s.n], s.λ + s.β*s.d[s.n .+ (1:s.m)],
        s.zl + s.β*s.dzl, s.zu + s.β*s.dzu,s),1)

    println("Fμ: $(Fμ)")
    println("Fμ⁺: $(Fμ⁺)")
    println("kkt error: $((Fμ⁺ <= Fμ))")
    return (Fμ⁺ <= s.opts.κF*Fμ)
end

function line_search(s::Solver)
    α_max!(s)
    s.l = 0
    status = false
    θ0 = θ(s.x + s.α_max*s.d[1:s.n],s)
    φ0 = barrier(s.x + s.α_max*s.d[1:s.n],s)

    while s.α > s.α_min
        if check_filter(θ(s.x + s.α*s.d[1:s.n],s),barrier(s.x + s.α*s.d[1:s.n],s),s)
            # case 1
            if (θ(s.x,s) < s.θ_min && switching_condition(s))
                if armijo(s.x + s.α*s.d[1:s.n],s)
                    status = true
                    break
                end
            # case 2
            else
                if sufficient_progress(s)
                    status = true
                    break
                end
            end
        end

        if s.l > 0 || θ(s.x + s.α_max*s.d[1:s.n],s) < θ(s.x,s)
            s.α *= 0.5
        else
            if second_order_correction(s)
                status = true
                break
            end
        end
        s.l += 1
    end

    return status
end

function restoration!(s::Solver)
    println("RESTORATION mode")
    status = false
    s.t = 0
    β_max!(s)

    while check_kkt_error(s)
        println("t: $(s.t)")
        if check_filter(θ(s.x + s.β*s.d[1:s.n],s),barrier(s.x + s.β*s.d[1:s.n],s),s::Solver)
            s.α = s.β
            s.αzl = s.β
            s.αzu = s.β
            println("KKT error reduction: success")
            status = true
            return
        else
            s.t += 1
            s.x .= s.x + s.β*s.d[1:s.n]
            s.λ .= s.λ + s.β*s.d[s.n .+ (1:s.m)]
            s.zl .= s.zl + s.β*s.dzl
            s.zu .= s.zu + s.β*s.dzu

            search_direction!(s)
            β_max!(s)
        end
    end

    if status
        return status
    else
        initialize_feasibility_restoration!(s)
        search_direction_restoration!(s)
        eval_Eμ_restoration(s.μ_res,s)
        α_max_restoration!(s)
        error("implement feasibility restoration")
    end
end

function update_μ!(s::Solver)
    s.μ = update_μ(s.μ,s.opts.κμ,s.opts.θμ,s.opts.ϵ_tol)
    return nothing
end

function update_τ!(s::Solver)
    s.τ = update_τ(s.μ,s.opts.τ_min)
    return nothing
end

function set_DR!(s::Solver)
    set_DR(s.DR,s.x,s.n)
    return nothing
end

function init_n!(s::Solver)
    s.c .= s.c_func(s.x)
    for i = 1:s.m
        s.n_res[i] = init_n(s.c[i],s.μ_res,s.opts.ρ)
    end
    return nothing
end

function init_p!(s::Solver)
    s.c .= s.c_func(s.x)
    for i = 1:s.m
        s.p_res[i] = init_p(s.n_res[i],s.c[i])
    end
    return nothing
end

function initialize_feasibility_restoration!(s::Solver)
    s.xR .= copy(s.x)
    s.λ .= 0.
    s.zl .= min.(s.opts.ρ,s.zl)
    s.zu .= min.(s.opts.ρ,s.zu)
    set_DR!(s)
    s.μ_res = max(s.μ,norm(s.c_func(s.x),Inf))
    s.ζ = sqrt(s.μ_res)
    init_n!(s)
    init_p!(s)
    s.zp_res .= s.μ_res./s.p_res
    s.zn_res .= s.μ_res./s.n_res
    return nothing
end

function search_direction_restoration!(s::Solver)
    ∇L(x) = s.∇c_func(x)'*s.λ
    s.W .= ForwardDiff.jacobian(∇L,s.x)
    s.Σl[CartesianIndex.((1:s.n)[s.xl_bool],(1:s.n)[s.xl_bool])] .= s.zl./((s.x - s.xl)[s.xl_bool])
    s.Σu[CartesianIndex.((1:s.n)[s.xu_bool],(1:s.n)[s.xu_bool])] .= s.zu./((s.xu - s.x)[s.xu_bool])

    s.c .= s.c_func(s.x)
    s.A .= s.∇c_func(s.x)

    s.H[1:s.n,1:s.n] .= (s.W + s.Σl + s.Σu)
    s.H[1:s.n,s.n .+ (1:s.m)] .= s.A'
    s.H[s.n .+ (1:s.m),1:s.n] .= s.A
    s.H[CartesianIndex.((s.n .+ (1:s.m)),(s.n .+ (1:s.m)))] .= -s.p_res./s.zp_res - s.n_res./s.zn_res

    s.h[1:s.n] .= s.ζ*s.DR'*s.DR*(s.x - s.xR) + s.A'*s.λ
    s.h[1:s.n][s.xl_bool] .-= s.μ_res./(s.x - s.xl)[s.xl_bool]
    s.h[1:s.n][s.xu_bool] .+= s.μ_res./(s.xu - s.x)[s.xu_bool]

    s.h[s.n .+ (1:s.m)] .= s.c - s.p_res + s.n_res + s.opts.ρ*(s.μ_res .- s.p_res)./s.zp_res + s.opts.ρ*(s.μ_res .- s.n_res)./s.zn_res

    s.d .= -s.H\s.h
    s.dzl .= -s.zl./((s.x - s.xl)[s.xl_bool]).*s.d[1:n][s.xl_bool] - s.zl + s.μ./((s.x - s.xl)[s.xl_bool])
    s.dzu .= s.zu./((s.xu - s.x)[s.xu_bool]).*s.d[1:n][s.xu_bool] - s.zu + s.μ./((s.xu - s.x)[s.xu_bool])

    s.dp .= (s.μ_res .+ s.p_res.*(s.λ + s.d[s.n .+ (1:s.m)]) - s.opts.ρ*s.p_res)./s.zp_res
    s.dn .= (s.μ_res .- s.n_res.*(s.λ + s.d[s.n .+ (1:s.m)]) - s.opts.ρ*s.n_res)./s.zn_res
    s.dzp .= s.μ_res./s.p_res - s.zp_res - (s.zp_res./s.p_res).*(s.dp)
    s.dzn .= s.μ_res./s.n_res - s.zn_res - (s.zn_res./s.n_res).*(s.dn)

    return nothing
end

function reset_z!(s::Solver)
    for i = 1:s.nl
        s.zl[i] = reset_z(s.zl[i],s.x[s.xl_bool][i],s.μ,s.opts.κΣ)
    end

    for i = 1:s.nu
        s.zu[i] = reset_z(s.zu[i],s.x[s.xu_bool][i],s.μ,s.opts.κΣ)
    end
    return nothing
end

function update!(s::Solver)
    if s.update == :nominal
        s.x .= s.x + s.α*s.d[1:s.n]
    elseif s.update == :soc
        s.x .= s.x + s.α_soc*s.d_soc[1:s.n]
        s.update = :nominal # reset update
    else
        error("update error")
    end
    s.λ .= s.λ + s.α*s.d[s.n .+ (1:s.m)]
    s.zl .= s.zl + s.αz*s.dzl
    s.zu .= s.zu + s.αz*s.dzu

    reset_z!(s)
    return nothing
end

function solve!(s::Solver)
    println("--solve initiated--")
    θ0 = θ(s.x,s)
    φ0 = barrier(s.x,s)
    println("φ0: $φ0, θ0: $θ0")

    # initialize filter
    push!(s.filter,(s.θ_max,Inf))

    println("Eμ0: $(eval_Eμ(0.0,s))")
    while eval_Eμ(0.0,s) > s.opts.ϵ_tol
        while eval_Eμ(s.μ,s) > s.opts.κϵ*s.μ
            search_direction!(s)
            if !line_search(s)
                restoration!(s)
            end
            augment_filter!(s)
            update!(s)

            s.k += 1
            if s.k > 100
                error("max iterations")
            end

            println("iteration (j,k): ($(s.j),$(s.k))")
            println("Eμ: $(eval_Eμ(s.μ,s))")
            println("θjk: $(θ(s.x,s)), φjk: $(barrier(s.x,s))\n")
        end
        s.k = 0
        s.j += 1

        update_μ!(s)
        update_τ!(s)

        empty!(s.filter)
        push!(s.filter,(s.θ_max,Inf))
    end
end
