function restoration!(s̄::Solver,s::Solver)
    println("~restoration phase~\n")
    if !kkt_error_reduction(s)
        # phase 2 solver
        initialize_restoration_solver!(s̄,s)

        # solve phase 2
        solve_restoration!(s̄,s,verbose=true)

        # update phase 1 solver
        update_phase1_solver!(s̄,s)
    else
        println("-KKT error reduction: success")
    end
    return nothing
end

function update_phase1_solver!(s̄::Solver,s::Solver)
    s.dx .= s̄.x[s.idx.x] - s.x
    s.dzL .= -s.zL./((s.x - s.xL)[s.xL_bool]).*s.d[s.idx.xL] - s.zL + s.μ./((s.x - s.xL)[s.xL_bool])
    s.dzU .= s.zU./((s.xU - s.x)[s.xU_bool]).*s.d[s.idx.xU] - s.zU + s.μ./((s.xU - s.x)[s.xU_bool])

    αz_max!(s)

    s.x .= s̄.x[s.idx.x]

    # project phase 2 solution on phase 1 bounds
    for i = 1:s.model.n
        s.x[i] = init_x0(s̄.x[i],s.xL[i],s.xU[i],s.opts.κ1,s.opts.κ2)
    end

    s.zL .+= s.αz*s.dzL
    s.zU .+= s.αz*s.dzU

    s.model.∇f_func!(s.∇f,s.x)
    s.model.∇c_func!(s.A,s.x)
    init_λ!(s.λ,s.H,s.h,s.d,s.zL,s.zU,s.∇f,s.A,s.model.n,s.model.m,s.xL_bool,s.xU_bool,s.opts.λ_max)

    return nothing
end

function solve_restoration!(s̄::Solver,s::Solver; verbose=false)
    # evaluate problem
    eval_iterate!(s̄)

    # initialize filter
    push!(s̄.filter,(s̄.θ_max,Inf))

    if verbose
        println("φ0: $(s̄.φ), θ0: $(s̄.θ)")
        println("Eμ0: $(eval_Eμ(0.0,s̄))\n")
    end

    while eval_Eμ(0.0,s̄) > s̄.opts.ϵ_tol
        while eval_Eμ(s̄.μ,s̄) > s̄.opts.κϵ*s̄.μ
            s̄.opts.relax_bnds ? relax_bnds!(s̄) : nothing
            if search_direction_restoration!(s̄,s)
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
                    restoration_reset!(s̄,s)
                else
                    augment_filter!(s̄)
                    update!(s̄)
                end
            end

            if check_filter(θ(s̄.x[s.idx.x],s),barrier(s̄.x[s.idx.x],s),s) && θ(s̄.x[s.idx.x],s) <= s̄.opts.κ_resto*s.θ
                # println("x: $(s̄.x[s.idx.x])")
                # println("p: $(s̄.x[s.model.n .+ (1:s.model.m)])")
                # println("n: $(s̄.x[(s.model.n+s.model.m) .+ (1:s.model.m)])")
                println("-restoration phase: success\n")
                return true
            end

            reset_z!(s̄)

            eval_iterate!(s̄)

            s̄.k += 1
            if s̄.k > s̄.opts.max_iter
                @warn "max iterations (restoration)"
                return
            end

            if verbose
                println("restoration iteration (j,k): ($(s̄.j),$(s̄.k))")
                # println("x: $(s̄.x)")
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

        update_restoration_objective!(s̄,s)
    end
    verbose ? println("<phase 2 complete>") : nothing
end

function restoration_reset!(s̄::Solver,s::Solver)
    s.model.c_func!(s.c,s̄.x[s.idx.x])

    # initialize p,n
    for i = 1:s.model.m
        s̄.x[s.model.n + s.model.m + i] = init_n(s.c[i],s̄.μ,s̄.opts.ρ)
    end

    for i = 1:s.model.m
        s̄.x[s.model.n + i] = init_p(s̄.x[s.model.n + s.model.m + i],s.c[i])
    end
    s̄.λ .= 0

    @warn "resetting restoration"

    return nothing
end

function RestorationSolver(s::Solver)
    opts = copy(s.opts)
    opts.λ_init_ls = false
    opts.relax_bnds = false

    n̄ = s.model.n + 2s.model.m
    m̄ = s.model.m

    x̄ = zeros(n̄)

    x̄L = zeros(n̄)
    x̄L[s.idx.x] = s.xL

    x̄U = Inf*ones(n̄)
    x̄U[s.idx.x] = s.xU

    f̄_func(x) = 0
    function ∇f̄_func(∇f,x)
        return nothing
    end
    function ∇²f̄_func(∇²f,x)
        return nothing
    end

    c̄_func(x) = zeros(m̄)
    function c̄_func(c,x)
        return nothing
    end
    ∇c̄_func(x) = zeros(m̄,n̄)
    function ∇c̄_func(∇c,x)
        return nothing
    end

    ∇²c̄λ_func(x,λ) = zeros(n̄,n̄)
    function ∇²c̄λ_func(∇²c̄λ,x,λ)
        return nothing
    end

    _model = Model(n̄,m̄,x̄L,x̄U,f̄_func,∇f̄_func,∇²f̄_func,c̄_func,∇c̄_func,∇²c̄λ_func)

    s̄ = Solver(x̄,_model,opts=opts)
    s̄.DR = spzeros(s.model.n,s.model.n)
    return s̄
end

function initialize_restoration_solver!(s̄::Solver,s::Solver)
    s̄.k = 0
    s̄.j = 0

    s̄.μ = max(s.μ,norm(s.c,Inf))
    s̄.τ = update_τ(s̄.μ,s̄.opts.τ_min)

    s̄.x[s.idx.x] = copy(s.x)

    # initialize p,n
    for i = 1:s.model.m
        s̄.x[s.model.n + s.model.m + i] = init_n(s.c[i],s̄.μ,s̄.opts.ρ)
    end

    for i = 1:s.model.m
        s̄.x[s.model.n + i] = init_p(s̄.x[s.model.n + s.model.m + i],s.c[i])
    end

    # initialize zL, zU, zp, zn
    for i = 1:s.nL
        s̄.zL[i] = min(s̄.opts.ρ,s.zL[i])
    end

    for i = 1:s.nU
        s̄.zU[i] = min(s̄.opts.ρ,s.zU[i])
    end

    s̄.zL[s.nL .+ (1:2s.model.m)] .= s̄.μ./s̄.x[s.model.n .+ (1:2s.model.m)]

    init_DR!(s̄.DR,s.x,s.model.n)

    s̄.restoration = true

    update_restoration_objective!(s̄,s)
    update_restoration_constraints!(s̄,s)
    empty!(s̄.filter)

    s̄.model.c_func!(s̄.c,s̄.x)
    s̄.θ = norm(s̄.c,1)
    s̄.θ_min = init_θ_min(s̄.θ)
    s̄.θ_max = init_θ_max(s̄.θ)

    return nothing
end

function update_restoration_objective!(s̄::Solver,s::Solver)
    ζ = sqrt(s̄.μ)
    DR = s̄.DR
    idx_pn = s.model.n .+ (1:2s.model.m)

    function f_func(x)
        s̄.opts.ρ*sum(x[idx_pn]) + 0.5*ζ*(x[s.idx.x] - s.x)'*DR'*DR*(x[s.idx.x] - s.x)
    end

    function ∇f_func!(∇f,x)
        ∇f[s.idx.x] .= ζ*DR'*DR*(x[s.idx.x] - s.x)
        ∇f[idx_pn] .= s̄.opts.ρ
        return nothing
    end

    function ∇²f_func!(∇²f,x)
        ∇²f[s.idx.x,s.idx.x] .= ζ*DR'*DR
        return nothing
    end

    s̄.model.f_func = f_func
    s̄.model.∇f_func! = ∇f_func!
    s̄.model.∇²f_func! = ∇²f_func!

    return nothing
end

function update_restoration_constraints!(s̄::Solver,s::Solver)
    function c_func!(c,x)
        s.model.c_func!(c,x[s.idx.x])
        c .-= x[s.model.n .+ (1:s.model.m)]
        c .+= x[(s.model.n+s.model.m) .+ (1:s.model.m)]
        return nothing
    end

    function ∇c_func!(∇c,x)
        s.model.∇c_func!(view(∇c,1:s.model.m,1:s.model.n),x[s.idx.x])
        ∇c[CartesianIndex.(1:s.model.m,s.model.n .+ (1:s.model.m))] .= -1.0
        ∇c[CartesianIndex.(1:s.model.m,s.model.n+s.model.m .+ (1:s.model.m))] .= 1.0
        return nothing
    end

    function ∇²cλ_func!(∇²cλ,x,λ)
        s.model.∇²cλ_func!(view(∇²cλ,s.idx.x,s.idx.x),x[s.idx.x],λ)
        return return nothing
    end

    s̄.model.c_func! = c_func!
    s̄.model.∇c_func! = ∇c_func!
    s̄.model.∇²cλ_func! = ∇²cλ_func!

    return nothing
end

function init_DR!(DR,xr,n)
    for i = 1:n
        DR[i,i] = min(1.0,1.0/abs(xr[i]))
    end
    return nothing
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

        s̄.δ[s.idx.x] .= s̄.δw
        s̄.δ[s̄.idx.λ] .= -s̄.δc
        s̄.d .= -(s̄.H + Diagonal(s̄.δ))\s̄.h

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
