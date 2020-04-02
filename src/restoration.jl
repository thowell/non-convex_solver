function restoration!(s::Solver)
    println("~restoration phase~\nj")
    if !kkt_error_reduction(s)
        # phase 2 solver
        s̄ = RestorationSolver(s)

        # solve phase 2
        solve_restoration!(s̄,s,verbose=true)

        # update phase 1 solver
        s.d[1:s.n] = s̄.x[1:s.n] - s.x
        s.d[(s.n+s.m) .+ (1:s.nL)] = -s.zL./((s.x - s.xL)[s.xL_bool]).*s.d[(1:s.n)[s.xL_bool]] - s.zL + s.μ./((s.x - s.xL)[s.xL_bool])
        s.d[(s.n+s.m+s.nL) .+ (1:s.nU)] = s.zU./((s.xU - s.x)[s.xU_bool]).*s.d[(1:s.n)[s.xU_bool]] - s.zU + s.μ./((s.xU - s.x)[s.xU_bool])

        αz_max!(s)

        s.x .= s̄.x[1:s.n]
        s.zL .+= s.αz*s.dzL
        s.zU .+= s.αz*s.dzU
        s.λ .= init_λ(s.zL,s.zU,∇f_func(s.x),∇c_func(s.x),s.n,s.m,s.xL_bool,s.xU_bool,s.opts.λ_max)
    else
        println("-KKT error reduction success")
    end
    return nothing
end

function solve_restoration!(s̄::Solver,s_ref::Solver; verbose=false)
    # evaluate problem
    eval_iterate!(s̄)

    # initialize filter
    push!(s̄.filter,(s̄.θ_max,Inf))

    if verbose
        println("θ_max: $(s̄.θ_max)")
        println("θ_min: $(s̄.θ_min)")
        println("φ0: $(s̄.φ), θ0: $(s̄.θ)")
        println("Eμ0: $(eval_Eμ(0.0,s̄))")
    end

    while eval_Eμ(0.0,s̄) > s̄.opts.ϵ_tol
        while eval_Eμ(s̄.μ,s̄) > s̄.opts.κϵ*s̄.μ
            if search_direction_restoration!(s̄,s_ref)
                s̄.small_search_direction_cnt += 1
                if s̄.small_search_direction_cnt == s̄.opts.small_search_direction_max
                    if s̄.μ < 0.1*s̄.opts.ϵ_tol
                        verbose ? println("<phase 2 complete>: small search direction") : nothing
                        return
                    end
                else
                    break
                end
                α_max!(s̄)
                αz_max!(s̄)
                augment_filter!(s̄)
                update!(s̄)
            else
                s̄.small_search_direction_cnt = 0

                if !line_search(s̄)
                    s_ref.c .= s_ref.c_func(s̄.x[1:s_ref.n])
                    # initialize p,n
                    for i = 1:s_ref.m
                        s̄.x[s_ref.n + s_ref.m + i] = init_n(s_ref.c[i],s̄.μ,s̄.opts.ρ)
                    end

                    for i = 1:s_ref.m
                        s̄.x[s_ref.n + i] = init_p(s̄.x[s_ref.n + s_ref.m + i],s_ref.c[i])
                    end
                    s̄.λ .= 0

                    @warn "resetting restoration"
                else
                    augment_filter!(s̄)
                    update!(s̄)
                end
            end

            if check_filter(θ(s̄.x[1:s_ref.n],s_ref),barrier(s̄.x[1:s_ref.n],s_ref),s_ref) && θ(s̄.x[1:s_ref.n],s_ref) <= s̄.opts.κ_resto*s_ref.θ
                println("x: $(s̄.x[1:s_ref.n])")
                println("p: $(s̄.x[s_ref.n .+ (1:s_ref.m)])")
                println("n: $(s̄.x[(s_ref.n+s_ref.m) .+ (1:s_ref.m)])")
                println("-restoration phase success\n")
                return true
            end

            eval_iterate!(s̄)

            s̄.k += 1
            if s̄.k > s̄.opts.max_iter
                # error("max iterations")
                return
            end

            if verbose
                println("restoration iteration (j,k): ($(s̄.j),$(s̄.k))")
                println("x: $(s̄.x)")
                println("θjk: $(θ(s̄.x,s̄)), φjk: $(barrier(s̄.x,s̄))")
                println("Eμ: $(eval_Eμ(s̄.μ,s̄))")
                println("α: $(s̄.α)\n")
            end
        end

        update_μ!(s̄)
        update_τ!(s̄)
        eval_barrier!(s̄)
        s̄.j += 1
        empty!(s̄.filter)
        push!(s̄.filter,(s̄.θ_max,Inf))

        if s̄.k == 0
            update_μ!(s̄)
            update_τ!(s̄)
            eval_barrier!(s̄)
            s̄.j += 1
            empty!(s̄.filter)
            push!(s̄.filter,(s̄.θ_max,Inf))
        end

        update_restoration_objective!(s̄,s_ref)
    end
    verbose ? println("<phase 2 complete>") : nothing
end

function RestorationSolver(s::Solver)
    s.c .= s.c_func(s.x)

    opts = copy(s.opts)
    opts.λ_init_ls = false
    opts.μ0 = max(s.μ,norm(s.c,Inf))

    n̄ = s.n + 2s.m
    m̄ = s.m

    x̄ = zeros(n̄)
    x̄[1:s.n] = copy(s.x)

    # initialize p,n
    for i = 1:s.m
        x̄[s.n + s.m + i] = init_n(s.c[i],opts.μ0,s.opts.ρ)
    end

    for i = 1:s.m
        x̄[s.n + i] = init_p(x̄[s.n + s.m + i],s.c[i])
    end

    x̄l = zeros(n̄)
    x̄l[1:s.n] = s.xL

    x̄u = Inf*ones(n̄)
    x̄u[1:s.n] = s.xU

    ζ = sqrt(opts.μ0)
    DR = init_DR(s.x,s.n)
    xR = copy(s.x)

    f_func(x) = s.opts.ρ*sum(x[s.n .+ (1:2s.m)]) + 0.5*ζ*(x[1:s.n] - xR)'*DR'*DR*(x[1:s.n] - xR)
    ∇f_func(x) = ForwardDiff.gradient(f_func,x)

    c_func(x) = s.c_func(x[1:s.n]) - x[s.n .+ (1:s.m)] + x[(s.n+s.m) .+ (1:s.m)]
    ∇c_func(x) = [s.∇c_func(x[1:s.n]) -I I]

    s̄ = Solver(x̄,n̄,m̄,x̄l,x̄u,f_func,c_func,∇f_func,∇c_func,opts=opts)

    # initialize zL, zU, zp, zn
    for i = 1:s.nL
        s̄.zL[i] = min(opts.ρ,s.zL[i])
    end

    for i = 1:s.nU
        s̄.zU[i] = min(opts.ρ,s.zU[i])
    end

    s̄.zL[s.nL .+ (1:2s.m)] .= opts.μ0./s̄.x[s.n .+ (1:2s.m)]

    s̄.DR = DR

    s̄.restoration = true

    return s̄
end

function update_restoration_objective!(s̄::Solver,s_ref::Solver)
    ζ = sqrt(s̄.μ)
    DR = s̄.DR
    function f_func(x)
        s̄.opts.ρ*sum(x[s_ref.n .+ (1:2s_ref.m)]) + 0.5*ζ*(x[1:s.n] - s_ref.x)'*DR'*DR*(x[1:s.n] - s_ref.x)
    end

    ∇f_func(x) = ForwardDiff.gradient(f_func,x)

    s̄.f_func = f_func
    s̄.∇f_func = ∇f_func
    return nothing
end

function init_DR(xr,n)
    DR = spzeros(n,n)
    for i = 1:n
        DR[i,i] = min(1.0,1.0/abs(xr[i]))
    end
    return DR
end

function init_n(c,μ,ρ)
    n = (μ - ρ*c)/(2.0*ρ) + sqrt(((μ-ρ*c)/(2.0*ρ))^2 + (μ*c)/(2.0*ρ))
    return n
end

function init_p(n,c)
    p = c + n
    return p
end

function search_direction_restoration!(s̄::Solver,s::Solver)
    if s.opts.kkt_solve == :unreduced
        kkt_hessian_unreduced!(s̄)
        kkt_gradient_unreduced!(s̄)

        flag = inertia_correction!(s̄)

        s̄.δ[1:s.n] .= s̄.δw
        s̄.δ[s.n + 2s.m .+ (1:s.m)] .= -s̄.δc
        s̄.d .= -(s̄.Hu + Diagonal(s̄.δ))\s̄.hu

        if flag
            iterative_refinement(s̄.d,s̄)
        end

        s̄.δw = 0.
        s̄.δc = 0.
    else
        error("KKT solve not implemented")
    end
    return small_search_direction(s)
end
