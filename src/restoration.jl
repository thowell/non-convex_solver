function RestorationSolver_slack(s::Solver)
    s.c .= s.c_func(s.x)

    opts = copy(s.opts)
    opts.λ_init_ls = true
    opts.μ0 = max(s.μ,norm(s.c,Inf))

    n̄ = s.n + 1
    m̄ = 1

    x̄ = zeros(n̄)
    x̄[1:s.n] .= s.x

    # initialize p,n
    # for i = 1:s.m
    #     x̄[s.n + s.m + i] = init_n(s.c[i],opts.μ0,s.opts.ρ)
    # end


    x̄[s.n + 1] = norm(s.c)^2

    δ = 0.0
    x̄l = zeros(n̄)
    x̄l[1:s.n] .= s.xl .+ δ

    x̄u = Inf*ones(n̄)
    x̄u[1:s.n] .= s.xu .- δ

    ζ = sqrt(opts.μ0)
    DR = init_DR(s.x,s.n)
    xR = copy(s.x)
    f_func(x) = s.opts.ρ*x[end] + 0.5*ζ*norm(DR*(x[1:s.n] - xR))^2
    ∇f_func(x) = ForwardDiff.gradient(f_func,x)
    # f_func(x) = s.opts.ρ*(x[s.n .+ (1:2s.m)]'*x[s.n .+ (1:2s.m)]) + 0.5*ζ*norm(DR*(x[1:s.n] - s.x))^2
    # ∇f_func(x) = [ζ*DR'*DR*(x[1:s.n] - s.x); 2.0*s.opts.ρ*x[s.n .+ (1:2s.m)]]

    c_func(x) = [x[s.n+1] - norm(s.c_func(x[1:s.n]))^2]
    c_func_d(x) = x[s.n+1] - norm(s.c_func(x[1:s.n]))^2

    ∇c_func(x) = Array(ForwardDiff.gradient(c_func_d,x)')

    rs = Solver(x̄,n̄,m̄,x̄l,x̄u,f_func,c_func,∇f_func,∇c_func,opts=opts)

    # initialize zl, zu, zp, zn
    for i = 1:s.nl
        rs.zl[i] = min(opts.ρ,s.zl[i])
    end

    for i = 1:s.nu
        rs.zu[i] = min(opts.ρ,s.zu[i])
    end

    rs.zl[s.nl + 1] = opts.μ0/rs.x[s.n + 1]
    # rs.zl[(s.nl+s.m) .+ (1:s.m)] .= opts.μ0./rs.x[(s.n+s.m) .+ (1:s.m)]

    rs.DR = DR

    rs.restoration = true

    return rs
end

function RestorationSolver(s::Solver)
    s.c .= s.c_func(s.x)

    opts = copy(s.opts)
    opts.λ_init_ls = false
    opts.μ0 = max(s.μ,norm(s.c,Inf))

    n̄ = s.n #+ 2s.m
    m̄ = 0

    x̄ = zeros(n̄)
    x̄[1:s.n] .= s.x

    # # initialize p,n
    # for i = 1:s.m
    #     x̄[s.n + s.m + i] = init_n(s.c[i],opts.μ0,s.opts.ρ)
    # end
    #
    # for i = 1:s.m
    #     x̄[s.n + i] = init_p(x̄[s.n + s.m + i],s.c[i])
    # end

    δ = 0.0
    x̄l = zeros(n̄)
    x̄l[1:s.n] .= s.xl .+ δ

    x̄u = Inf*ones(n̄)
    x̄u[1:s.n] .= s.xu .- δ

    ζ = sqrt(opts.μ0)
    DR = init_DR(s.x,s.n)
    xR = copy(s.x)
    f_func(x) = s.opts.ρ*s.c_func(x)'*s.c_func(x) + 0.5*ζ*norm(DR*(x[1:s.n] - xR))^2
    ∇f_func(x) = ForwardDiff.gradient(f_func,x)#[ζ*DR'*DR*(x[1:s.n] - xR); s.opts.ρ*ones(2s.m)]
    # f_func(x) = s.opts.ρ*(x[s.n .+ (1:2s.m)]'*x[s.n .+ (1:2s.m)]) + 0.5*ζ*norm(DR*(x[1:s.n] - s.x))^2
    # ∇f_func(x) = [ζ*DR'*DR*(x[1:s.n] - s.x); 2.0*s.opts.ρ*x[s.n .+ (1:2s.m)]]

    c_func(x) = 0.#s.c_func(x[1:s.n]) - x[s.n .+ (1:s.m)] + x[(s.n+s.m) .+ (1:s.m)]
    ∇c_func(x) = zeros(0,n)#[s.∇c_func(x[1:s.n]) -I I]

    rs = Solver(x̄,n̄,m̄,x̄l,x̄u,f_func,c_func,∇f_func,∇c_func,opts=opts)

    # # initialize zl, zu, zp, zn
    # for i = 1:s.nl
    #     rs.zl[i] = min(opts.ρ,s.zl[i])
    # end
    #
    # for i = 1:s.nu
    #     rs.zu[i] = min(opts.ρ,s.zu[i])
    # end

    # rs.zl[s.nl .+ (1:2s.m)] .= opts.μ0./rs.x[s.n .+ (1:2s.m)]
    # rs.zl[(s.nl+s.m) .+ (1:s.m)] .= opts.μ0./rs.x[(s.n+s.m) .+ (1:s.m)]

    rs.DR = DR

    rs.restoration = true

    return rs
end

function RestorationSolver_l1(s::Solver)
    s.c .= s.c_func(s.x)

    opts = copy(s.opts)
    opts.λ_init_ls = false
    opts.μ0 = max(s.μ,norm(s.c,Inf))

    n̄ = s.n + 2s.m
    m̄ = s.m

    x̄ = zeros(n̄)
    x̄[1:s.n] .= s.x

    # initialize p,n
    for i = 1:s.m
        x̄[s.n + s.m + i] = init_n(s.c[i],opts.μ0,s.opts.ρ)
    end

    for i = 1:s.m
        x̄[s.n + i] = init_p(x̄[s.n + s.m + i],s.c[i])
    end

    x̄l = zeros(n̄)
    x̄l[1:s.n] .= s.xl

    x̄u = Inf*ones(n̄)
    x̄u[1:s.n] .= s.xu

    ζ = sqrt(opts.μ0)
    DR = init_DR(s.x,s.n)
    xR = copy(s.x)

    f_func(x) = s.opts.ρ*sum(x[s.n .+ (1:2s.m)]) + 0.5*ζ*norm(DR*(x[1:s.n] - xR))^2 + 2.0*s.opts.ρ*x[end]
    ∇f_func(x) = ForwardDiff.gradient(f_func,x)#[ζ*DR'*DR*(x[1:s.n] - s.x); s.opts.ρ*ones(2s.m)]

    c_func(x) = s.c_func(x[1:s.n]) - x[s.n .+ (1:s.m)] + x[(s.n+s.m) .+ (1:s.m)]
    ∇c_func(x) = [s.∇c_func(x[1:s.n]) -I I]

    rs = Solver(x̄,n̄,m̄,x̄l,x̄u,f_func,c_func,∇f_func,∇c_func,opts=opts)

    # initialize zl, zu, zp, zn
    for i = 1:s.nl
        rs.zl[i] = min(opts.ρ,s.zl[i])
    end

    for i = 1:s.nu
        rs.zu[i] = min(opts.ρ,s.zu[i])
    end

    rs.zl[s.nl .+ (1:2s.m)] .= opts.μ0./rs.x[s.n .+ (1:2s.m)]

    rs.DR = DR

    rs.restoration = true

    return rs
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

function restoration!(s::Solver)
    println("RESTORATION mode")
    status = false
    s.t = 0
    # β_max!(s)

    # while check_kkt_error(s)
    #     println("t: $(s.t)")
    #     if check_filter(θ(s.x + s.β*s.d[1:s.n],s),barrier(s.x + s.β*s.d[1:s.n],s),s::Solver)
    #         s.α = s.β
    #         s.αz = s.β
    #         println("KKT error reduction: success")
    #         status = true
    #         return
    #     else
    #         s.t += 1
    #         s.x .= s.x + s.β*s.d[1:s.n]
    #         s.λ .= s.λ + s.β*s.d[s.n .+ (1:s.m)]
    #         s.zl .= s.zl + s.β*s.dzl
    #         s.zu .= s.zu + s.β*s.dzu
    #
    #         search_direction!(s)
    #         β_max!(s)
    #     end
    # end

    if status
        return status
    else
        rs = RestorationSolver_slack(s)
        solve!(rs,s)

        println("restoration update result")
        println("x0: $(s.x)")
        dx = rs.x[1:s.n] - s.x

        s.dzl .= -s.zl./((s.x - s.xl)[s.xl_bool]).*dx[s.xl_bool] - s.zl + s.μ./((s.x - s.xl)[s.xl_bool])
        s.dzu .= s.zu./((s.xu - s.x)[s.xu_bool]).*dx[s.xu_bool] - s.zu + s.μ./((s.xu - s.x)[s.xu_bool])
        s.x .= rs.x[1:s.n]

        s.αz = 1.0
        while !fraction_to_boundary(s.zl,s.dzl,s.αz,s.τ)
            s.αz *= 0.5
        end

        while !fraction_to_boundary(s.zu,s.dzu,s.αz,s.τ)
            s.αz *= 0.5
        end

        s.zl .= s.zl + s.αz*s.dzl
        s.zu .= s.zu + s.αz*s.dzu
        s.λ .= init_λ(s.zl,s.zu,s.∇f_func(s.x),s.∇c_func(s.x),s.n,s.m,s.xl_bool,s.xu_bool,s.opts.λ_max)

        println("x: $(s.x)")
        println("φ: $(barrier(s.x,s))")
        println("θ: $(θ(s.x,s))")
        # error("restoration stop")
        return true
    end
end

function update_restoration_objective!(s::Solver,s_ref::Solver)
    ζ = sqrt(s.μ)
    DR = s.DR
    function f_func(x)
        println("updated!- ζ: $(ζ)")
        # s.opts.ρ*sum(x[s_ref.n .+ (1:2s_ref.m)]) + 0.5*ζ*norm(DR*(x[1:s_ref.n] - s_ref.x))^2 + 2.0*s.opts.ρ*x[end]
        # s.opts.ρ*s_ref.c_func(x)'*s_ref.c_func(x) + 0.5*ζ*norm(DR*(x[1:s.n] - s_ref.x))^2
        s.opts.ρ*x[s_ref.n+1] + 0.5*ζ*norm(DR*(x[1:s_ref.n] - s_ref.x))^2
    end

    ∇f_func(x) = ForwardDiff.gradient(f_func,x)#[ζ*DR'*DR*(x[1:s.n] - s.x); s.opts.ρ*ones(2s.m)]

    s.f_func = f_func
    s.∇f_func = ∇f_func
    return nothing
end

function augment_filter_restoration!(x⁺,s::Solver)
    θ⁺ = θ(x⁺,s)
    φ⁺ = barrier(x⁺,s)

    if !switching_condition(s) || !armijo(x⁺,s)
        add_to_filter!((θ⁺,φ⁺),s)
    end

    return nothing
end

function solve!(s::Solver,s_ref::Solver)
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
                error("restoration restoration")
            else
                augment_filter!(s)
                update!(s)
            end

            s.k += 1
            if s.k > s.opts.max_iter
                error("max iterations")
            end

            println("restoration iteration (j,k): ($(s.j),$(s.k))")
            println("x: $(s.x)")
            println("Eμ: $(eval_Eμ(s.μ,s))")
            println("θjk: $(θ(s.x,s)), φjk: $(barrier(s.x,s))\n")

            if check_filter(θ(s.x[1:s_ref.n],s_ref),barrier(s.x[1:s_ref.n],s_ref),s_ref) && θ(s.x[1:s_ref.n],s_ref) <= s.opts.κ_resto*θ(s_ref.x,s_ref)
                return true
            end
        end
        s.k = 0
        s.j += 1

        update_μ!(s)
        update_τ!(s)

        update_restoration_objective!(s,s_ref)

        empty!(s.filter)
        push!(s.filter,(s.θ_max,Inf))
    end
end
