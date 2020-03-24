function RestorationSolver(s::Solver)
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
        x̄[s.n + i] = init_n(s.c[i],opts.μ0,s.opts.ρ)
    end

    for i = 1:s.m
        x̄[(s.n + s.m) + i] = init_p(x̄[s.n + i],s.c[i])
    end

    x̄l = zeros(n̄)
    x̄l[1:s.n] .= s.xl

    x̄u = Inf*ones(n̄)
    x̄u[1:s.n] .= s.xu

    ζ = sqrt(opts.μ0)
    DR = init_DR(s.x,s.n)
    f_func(x) = s.opts.ρ*(sum(x[s.n .+ (1:2s.m)])) + 0.5*ζ*norm(DR*(x[1:s.n] - s.x))^2
    ∇f_func(x) = [ζ*DR'*DR*(x[1:s.n] - s.x); s.opts.ρ*ones(2s.m)]

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

    rs.zl[s.nl .+ (1:s.m)] .= opts.μ0./rs.x[s.n .+ (1:s.m)]
    rs.zl[(s.nl+s.m) .+ (1:s.m)] .= opts.μ0./rs.x[(s.n+s.m) .+ (1:s.m)]

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
        rs = RestorationSolver(s)
        solve_restoration!(rs,s)

        println("restoration update result")
        println("x0: $(s.x)")
        dx = rs.x[1:s.n] - s.x

        s.dzl .= -s.zl./((s.x - s.xl)[s.xl_bool]).*dx[s.xl_bool] - s.zl + s.μ./((s.x - s.xl)[s.xl_bool])
        s.dzu .= s.zu./((s.xu - s.x)[s.xu_bool]).*dx[s.xu_bool] - s.zu + s.μ./((s.xu - s.x)[s.xu_bool])
        s.x .= rs.x[1:s.n]

        s.αz = 1.0
        while !fraction_to_boundary(s.zl,s.dzl,s.αz,s.τ)
            s.αz *= 0.5
            # println("αzl = $(s.αz)")
            # if s.αz < s.α_min
            #     error("αzl < α_min")
            # end
        end

        while !fraction_to_boundary(s.zu,s.dzu,s.αz,s.τ)
            s.αz *= 0.5
            # println("αzu = $(s.αz)")
            # if s.αz < s.α_min
            #     error("αzu < α_min")
            # end
        end
        s.zl .= s.zl + s.αz*s.dzl
        s.zu .= s.zu + s.αz*s.dzu
        s.λ .= init_λ(s.zl,s.zu,s.∇f_func(s.x),s.∇c_func(s.x),s.n,s.m,s.xl_bool,s.xu_bool,s.opts.λ_max)

        println("x: $(s.x)")
        println("φ: $(barrier(s.x,s))")
        println("θ: $(θ(s.x,s))")
        # error("restoration success")
        return true
    end
end

function update_restoration_objective!(s::Solver,n,xR)
    if s.restoration
        ζ = sqrt(s.μ)
        f_func(x) = s.opts.ρ*(sum(x[n .+ (1:2s.m)])) + 0.5*ζ*norm(s.DR*(x[1:n] - xR))^2
        ∇f_func(x) = [ζ*s.DR'*s.DR*(x[1:n] - xR); s.opts.ρ*ones(2s.m)]
    end
    return nothing
end

function solve_restoration!(s::Solver,sR::Solver)
    println("-- restoration solve initiated--")
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
                error("restoration solve called restoration mode")
            end
            augment_filter!(s)
            update!(s)

            s.k += 1
            if s.k > 1000
                error("max iterations")
            end

            if check_filter(θ(s.x[1:sR.n],sR),barrier(s.x[1:sR.n],sR),sR) && θ(s.x[1:sR.n],sR) <= s.opts.κ_resto*θ(sR.x,sR)
                return true
            end

            println("iteration (j,k): ($(s.j),$(s.k))")
            println("Eμ: $(eval_Eμ(s.μ,s))")
            println("θjk: $(θ(s.x,s)), φjk: $(barrier(s.x,s))\n")
        end
        s.k = 0
        s.j += 1

        update_μ!(s)
        update_τ!(s)

        update_restoration_objective!(s,sR.n,sR.x)

        empty!(s.filter)
        push!(s.filter,(s.θ_max,Inf))
    end
end

function augment_filter_restoration!(x⁺,s::Solver)

    θ⁺ = θ(x⁺,s)
    φ⁺ = barrier(x⁺,s)

    if !switching_condition(s) || !armijo(x⁺,s)
        add_to_filter!((θ⁺,φ⁺),s)
    end

    return nothing
end
