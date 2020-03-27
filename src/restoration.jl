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
    f_func(x) = s.opts.ρ*x[s.n+1] + 0.5*ζ*norm(DR*(x[1:s.n] - xR))^2
    ∇f_func(x) = [ζ*DR'*DR*(x[1:s.n] - xR); 1.0*s.opts.ρ]
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
    f_func_(x) = s.opts.ρ*s.c_func(x)'*s.c_func(x) + 0.5*ζ*norm(DR*(x[1:s.n] - xR))^2

    ∇f_func(x) = ForwardDiff.gradient(f_func_,x)#[ζ*DR'*DR*(x[1:s.n] - xR); s.opts.ρ*ones(2s.m)]
    # f_func(x) = s.opts.ρ*(x[s.n .+ (1:2s.m)]'*x[s.n .+ (1:2s.m)]) + 0.5*ζ*norm(DR*(x[1:s.n] - s.x))^2
    # ∇f_func(x) = [ζ*DR'*DR*(x[1:s.n] - s.x); 2.0*s.opts.ρ*x[s.n .+ (1:2s.m)]]

    c_func(x) = 0.#s.c_func(x[1:s.n]) - x[s.n .+ (1:s.m)] + x[(s.n+s.m) .+ (1:s.m)]
    ∇c_func(x) = zeros(0,n)#[s.∇c_func(x[1:s.n]) -I I]

    rs = Solver(x̄,n̄,m̄,x̄l,x̄u,f_func,c_func,∇f_func,∇c_func,opts=opts)

    # initialize zl, zu, zp, zn
    for i = 1:s.nl
        rs.zl[i] = min(opts.ρ,s.zl[i])
    end

    for i = 1:s.nu
        rs.zu[i] = min(opts.ρ,s.zu[i])
    end

    # rs.zl[s.nl .+ (1:2s.m)] .= opts.μ0./rs.x[s.n .+ (1:2s.m)]
    # rs.zl[(s.nl+s.m) .+ (1:s.m)] .= opts.μ0./rs.x[(s.n+s.m) .+ (1:s.m)]

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
        # error("restoration stop")
        return true
    end
end

function update_restoration_objective!(s::Solver,n,m,xR)
    ζ = sqrt(s.μ)
    f_func(x) = s.opts.ρ*x[n +1] + 0.5*ζ*norm(DR*(x[1:n] - xR))^2
    ∇f_func(x) = [ζ*DR'*DR*(x[1:n] - xR); s.opts.ρ]

    s.f_func = f_func
    s.∇f_func = ∇f_func
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
        while eval_Eμ(s.μ,s) > s.opts.κϵ*s.μ || s.k == 0
            search_direction!(s)
            if !line_search(s)
                error("restoration restoration")
                s.c .= sR.c_func(s.x[1:sR.n])

                # # initialize p,n
                # for i = 1:s.m
                #     s.x[sR.n + s.m + i] = init_n(s.c[i],s.μ,s.opts.ρ)
                # end
                #
                # for i = 1:s.m
                #     s.x[sR.n + i] = init_p(s.x[sR.n + s.m + i],s.c[i])
                # end

            end
            augment_filter!(s)
            update!(s)

            s.k += 1
            if s.k > s.opts.max_iter
                error("max iterations")
            end

            if check_filter(θ(s.x[1:sR.n],sR),barrier(s.x[1:sR.n],sR),sR) && θ(s.x[1:sR.n],sR) <= s.opts.κ_resto*θ(sR.x,sR)
                return true
            end

            println("iteration (j,k): ($(s.j),$(s.k))")
            println("x: $(s.x[1:sR.n])")
            # println("p: $(s.x[sR.n .+ (1:s.m)])")
            # println("n: $(s.x[(sR.n+s.m) .+ (1:s.m)])")
            println("Eμ: $(eval_Eμ(s.μ,s))")
            println("θjk: $(θ(s.x,s)), φjk: $(barrier(s.x,s))\n")
            println("x: $(s.x[1:sR.n])")
        end
        s.k = 0
        s.j += 1

        update_μ!(s)
        update_τ!(s)

        update_restoration_objective!(s,sR.n,sR.m,copy(sR.x))

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

function search_direction_restoration!(sr::Solver,s::Solver)
    H̄ = spzeros(s.n+s.m,s.n+s.m)
    h̄ = zeros(s.n+s.m)

    x = sr.x[1:s.n]
    p = sr.x[s.n .+ (1:s.m)]
    n = sr.x[(s.n+s.m) .+ (1:s.m)]
    zl = sr.zl[1:s.nl]
    zu = sr.zu[1:s.nu]
    zp = sr.zl[s.nl .+ (1:s.m)]
    zn = sr.zl[(s.nl + s.m) .+ (1:s.m)]

    ∇L(z) = s.∇c_func(z[1:s.n])'*sr.λ
    W̄ = ForwardDiff.jacobian(∇L,x)
    s.Σl[CartesianIndex.((1:s.n)[s.xl_bool],(1:s.n)[s.xl_bool])] .= zl./((x - s.xl)[s.xl_bool])
    s.Σu[CartesianIndex.((1:s.n)[s.xu_bool],(1:s.n)[s.xu_bool])] .= zu./((s.xu - x)[s.xu_bool])
    c = s.c_func(x)
    A = s.∇c_func(x)
    ζ = sqrt(sr.μ)

    H̄[1:s.n,1:s.n] .= (W̄ + ζ*sr.DR'*sr.DR + s.Σl + s.Σu)
    H̄[1:s.n,s.n .+ (1:s.m)] .= A'
    H̄[s.n .+ (1:s.m),1:s.n] .= A
    H̄[s.n .+ (1:s.m),s.n .+ (1:s.m)] .= -p./zp - n./zn

    h̄[1:s.n] .= ζ*sr.DR'*sr.DR*(x - s.x) + A'*sr.λ
    h̄[1:s.n][s.xl_bool] .-= sr.μ./(x - s.xl)[s.xl_bool]
    h̄[1:s.n][s.xu_bool] .+= sr.μ./(s.xu - x)[s.xu_bool]
    h̄[s.n .+ (1:s.m)] .= c - p + n + s.opts.ρ*(sr.μ .- p)./zp + s.opts.ρ*(sr.μ .- n)./zn

    d = -H̄\h̄

    dx = d[1:s.n]
    dλ = d[s.n .+ (1:s.m)]

    dp = (sr.μ .+ p.*(sr.λ + dλ) - s.opts.ρ*p)./zp
    dn = (sr.μ .- n.*(sr.λ + dλ) - s.opts.ρ*n)./zn

    dzl = -zl./((x - s.xl)[s.xl_bool]).*dx[s.xl_bool] - zl + sr.μ./((x - s.xl)[s.xl_bool])
    dzu = zu./((s.xu - x)[s.xu_bool]).*dx[s.xu_bool] - zu + sr.μ./((s.xu - x)[s.xu_bool])
    dzp = sr.μ./p - zp - zp./p.*dp
    dzn = sr.μ./n - zn - zn./n.*dn

    _d = [dx;dp;dn;dλ]

    sr.d .= _d
    sr.dzl .= [dzl;dzp;dzn]
    sr.dzu .= dzu

    return nothing
end
