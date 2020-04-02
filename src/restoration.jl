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
        # for i = 1:s.n
        #     s.x[i] = init_x0(s.x[i],s.xL[i],s.xU[i],s.opts.κ1,s.opts.κ2)
        # end
        s.zL .+= s.αz*s.dzL
        s.zU .+= s.αz*s.dzU
        s.λ .= init_λ(s.zL,s.zU,∇f_func(s.x),∇c_func(s.x),s.n,s.m,s.xL_bool,s.xU_bool,s.opts.λ_max)
        s.H[s.n .+ (1:s.m),s.n .+ (1:s.m)] .= 0.
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
            elseif !line_search(s̄)
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
                s̄.small_search_direction_cnt = 0
            else
                augment_filter!(s̄)
                update!(s̄)
                s̄.small_search_direction_cnt = 0
            end

            if check_filter(θ(s̄.x[1:s_ref.n],s_ref),barrier(s̄.x[1:s_ref.n],s_ref),s_ref) && θ(s̄.x[1:s_ref.n],s_ref) <= s̄.opts.κ_resto*s_ref.θ
                println("x: $(s̄.x[1:s_ref.n])")
                println("p: $(s̄.x[s_ref.n .+ (1:s_ref.m)])")
                println("n: $(s̄.x[(s_ref.n+s_ref.m) .+ (1:s_ref.m)])")
                println("-restoration phase success\n")
                return true
            end
            # reset_z!(s̄)

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
        # println("updated!- ζ: $(ζ)")
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



# symmetric KKT system
function kkt_hessian_symmetric_restoration!(s̄::Solver,s::Solver)
    s.W .= s̄.W[1:s.n,1:s.n]
    s̄.ΣL[CartesianIndex.((1:s̄.n)[s̄.xL_bool],(1:s̄.n)[s̄.xL_bool])] .= s̄.zL./((s̄.x - s̄.xL)[s̄.xL_bool])
    s̄.ΣU[CartesianIndex.((1:s̄.n)[s̄.xU_bool],(1:s̄.n)[s̄.xU_bool])] .= s̄.zU./((s̄.xU - s̄.x)[s̄.xU_bool])
    s.ΣL .= s̄.ΣL[1:s.n,1:s.n]
    s.ΣU .= s̄.ΣU[1:s.n,1:s.n]

    # Σp = Array(s̄.ΣL[s.n .+ (1:s.m),s.n .+ (1:s.m)])
    # Σn = Array(s̄.ΣU[(s.n+s.m) .+ (1:s.m),(s.n+s.m) .+ (1:s.m)])

    p = s̄.x[s.n .+ (1:s.m)]
    n = s̄.x[(s.n + s.m) .+ (1:s.m)]

    zL = s̄.zL[1:s.nL]
    zp = s̄.zL[s.nL .+ (1:s.m)]
    zn = s̄.zL[(s.nL+s.m) .+ (1:s.m)]

    s.A .= s̄.A[1:s.m,1:s.n]

    s.H[1:s.n,1:s.n] .= s.W + s.ΣL + s.ΣU
    s.H[1:s.n,s.n .+ (1:s.m)] .= s.A'
    s.H[s.n .+ (1:s.m),1:s.n] .= s.A
    s.H[s.n .+ (1:s.m),s.n .+ (1:s.m)] = -Diagonal(p./zp) - Diagonal(n./zn)

    return nothing
end

function kkt_gradient_symmetric_restoration!(s̄::Solver,s::Solver)
    s.∇φ .= s̄.∇φ[1:s.n]
    s.A .= s̄.A[1:s.m,1:s.n]

    p = s̄.x[s.n .+ (1:s.m)]
    n = s̄.x[(s.n + s.m) .+ (1:s.m)]

    λ = s̄.λ

    ρ = s̄.opts.ρ
    μ = s̄.μ

    zL = s̄.zL[1:s.nL]
    zp = s̄.zL[s.nL .+ (1:s.m)]
    zn = s̄.zL[(s.nL+s.m) .+ (1:s.m)]

    s.h[1:s.n] = s.∇φ + s.A'*λ
    s.h[s.n .+ (1:s.m)] = s̄.c + ρ*Diagonal(zp)\(μ*ones(s.m) - p) + ρ*Diagonal(zn)\(μ*ones(s.m) - n)

    return nothing
end

function search_direction_symmetric_restoration!(s̄::Solver,s::Solver)
    kkt_hessian_symmetric_restoration!(s̄,s)
    kkt_gradient_symmetric_restoration!(s̄,s)

    flag = inertia_correction_hsl!(s.H,s,true)

    x = s̄.x[(1:s.n)]
    p = s̄.x[s.n .+ (1:s.m)]
    n = s̄.x[(s.n + s.m) .+ (1:s.m)]

    λ = s̄.λ

    zL = s̄.zL[1:s.nL]
    zp = s̄.zL[s.nL .+ (1:s.m)]
    zn = s̄.zL[(s.nL+s.m) .+ (1:s.m)]

    zU = s̄.zU[1:s.nU]

    μ = s̄.μ
    ρ = s̄.opts.ρ

    Σp = Diagonal(zp./p)
    Σn = Diagonal(zn./n)

    # dx, dλ
    LBL = Ma57(s.H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))
    ma57_factorize(LBL)
    s̄.d[[(1:s.n)...,((s.n+2s.m) .+ (1:s.m))...]] = ma57_solve(LBL, -s.h)

    # s̄.d[[(1:s.n)...,((s.n+2s.m) .+ (1:s.m))...]] = -Symmetric(s.H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))\s.h

    dx = s̄.d[(1:s.n)]
    dλ = s̄.d[((s.n+2s.m) .+ (1:s.m))]

    # dp
    s̄.d[s.n .+ (1:s.m)] = Diagonal(zp)\(μ*ones(s.m) + Diagonal(p)*(λ + dλ) - ρ*p)
    dp = s̄.d[s.n .+ (1:s.m)]

    # dn
    s̄.d[(s.n+s.m) .+ (1:s.m)] = Diagonal(zn)\(μ*ones(s.m) - Diagonal(n)*(λ + dλ) - ρ*n)
    dn = s̄.d[(s.n+s.m) .+ (1:s.m)]

    # dzL
    s̄.d[(s.n + 2s.m + s.m) .+ (1:s.nL)] = -zL./((x - s.xL)[s.xL_bool]).*dx[s.xL_bool] - zL + μ./((x - s.xL)[s.xL_bool])
    dzL = s̄.d[(s.n + 2s.m + s.m) .+ (1:s.nL)]

    #dzU
    s̄.d[(s.n + 2s.m + s.m + s.nL + s.m + s.m) .+ (1:s.nU)] = zU./((s.xU - x)[s.xU_bool]).*dx[s.xU_bool] - zU + μ./((s.xU - x)[s.xU_bool])
    dzU = s̄.d[(s.n + 2s.m + s.m + s.nL + s.m + s.m) .+ (1:s.nU)]

    # dzp
    s̄.d[(s.n + 2s.m + s.m + s.nL) .+ (1:s.m)] = μ*Diagonal(p)\ones(s.m) - zp - Σp*dp
    dzp = s̄.d[(s.n + 2s.m + s.m + s.nL) .+ (1:s.m)]

    # dzn
    s̄.d[(s.n + 2s.m + s.m + s.nL + s.m) .+ (1:s.m)] = μ*Diagonal(n)*ones(s.m) - zn - Σn*dn
    dzn = s̄.d[(s.n + 2s.m + s.m + s.nL + s.m) .+ (1:s.m)]

    if flag
        # kkt_hessian_unreduced!(s̄)
        # kkt_gradient_unreduced!(s̄)
        # iterative_refinement(s̄.d,s̄.Hu,
        #     [s.δw*ones(s.n);zeros(2s.m);-s.δc*ones(s.m);zeros(s̄.nL+s̄.nU)],-s̄.hu,s̄.n,s̄.m,
        #     max_iter=s.opts.max_iterative_refinement,
        #     ϵ=s.opts.ϵ_iterative_refinement)
        iterative_refinement(s̄.d,s̄,s)
    end
    s.δw = 0.
    s.δc = 0.

    return small_search_direction(s̄)
end

function search_direction_restoration!(s̄::Solver,s::Solver)
    if s̄.opts.kkt_solve == :symmetric
        search_direction_symmetric_restoration!(s̄,s)
    elseif s.opts.kkt_solve == :unreduced
        kkt_hessian_unreduced!(s̄)
        kkt_gradient_unreduced!(s̄)

        flag = inertia_correction_hsl!(s.H,s)

        δ = [s.δw*ones(s.n);zeros(2s.m);-s.δc*ones(s.m);zeros(s̄.nL+s̄.nU)]
        s̄.d .= -(s̄.Hu + Diagonal(δ))\s̄.hu

        if flag
            s̄.δw = s.δw
            s̄.δc = s.δc
            iterative_refinement(s̄.d,δ,s̄)
        end

        s.δw = 0.
        s.δc = 0.
    else
        error("KKT solve not implemented")
    end
    return small_search_direction(s)
end
