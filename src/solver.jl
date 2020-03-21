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

    Solver(x,xl,xu,xl_bool,xu_bool,x_soc,λ,zl,zu,n,nl,nu,m,f_func,∇f_func,c_func,
        ∇c_func,H,h,W,Σl,Σu,A,∇f,∇φ,∇L,c,c_soc,d,d_soc,dzl,dzu,μ,α,αz,α_max,α_min,
        α_soc,update,τ,δw,δc,θ,θ_min,θ_max,θ_soc,sd,sc,filter,j,k,l,p,opts)
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
    s.dzu = s.zu./((s.xu - s.x)[s.xu_bool]).*s.d[1:n][s.xu_bool] - s.zu + s.μ./((s.xu - s.x)[s.xu_bool])
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

    # s.αz = 1.0
    # while !fraction_to_boundary([s.zl;s.zu],[s.dzl;s.dzu],s.αz,s.τ)
    #     s.αz *= 0.5
    #     println("αz = $(s.αz)")
    #     if s.αz < s.α_min
    #         error("αz < α_min")
    #     end
    # end

    αzl = 1.0
    while !fraction_to_boundary(s.zl,s.dzl,αzl,s.τ)
        αzl *= 0.5
        println("αzl = $(αzl)")
        if αzl < s.α_min
            error("αzl < α_min")
        end
    end

    αzu = 1.0
    while !fraction_to_boundary(s.zu,s.dzu,αzu,s.τ)
        αzu *= 0.5
        println("αzu = $(αzu)")
        if αzu < s.α_min
            error("αzu < α_min")
        end
    end

    s.αz = min(αzl,αzu)

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
    for f in s.filter
        if θ < f[1] || φ < f[2]
            return true
        end
    end
    return false
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
        push!(s.filter,(θ⁺,φ⁺))
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
        return true
    else
        return false
    end
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

    if !status
        error("implement feasibility restoration")
    end

    augment_filter!(s)

    return true
end

function update_μ!(s::Solver)
    s.μ = update_μ(s.μ,s.opts.κμ,s.opts.θμ,s.opts.ϵ_tol)
    return nothing
end

function update_τ!(s::Solver)
    s.τ = update_τ(s.μ,s.opts.τ_min)
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
            flag = line_search(s)
            update!(s)
            println("line search: $flag")
            println("Eμ: $(eval_Eμ(s.μ,s))")

            s.k += 1
            if s.k > 100
                error("hi")
            end
        end
        s.k = 0
        s.j += 1

        update_μ!(s)
        update_τ!(s)

        empty!(s.filter)
        push!(s.filter,(s.θ_max,Inf))
    end
end
