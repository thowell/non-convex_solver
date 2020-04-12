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
    # for i = 1:s.model.n
    #     s.x[i] = init_x0(s̄.x[i],s.xL[i],s.xU[i],s.opts.κ1,s.opts.κ2)
    # end

    s.zL .+= s.αz*s.dzL
    s.zU .+= s.αz*s.dzU

    s.model.∇f_func!(s.∇f,s.x)
    s.model.∇c_func!(s.A,s.x)
    init_λ!(s.λ,s.H_sym,s.h_sym,s.d,s.zL,s.zU,s.∇f,s.A,s.model.n,s.model.m,s.xL_bool,s.xU_bool,s.opts.λ_max)

    return nothing
end

function solve_restoration!(s̄::Solver,s::Solver; verbose=false)
    # evaluate problem
    eval_iterate!(s̄)

    # initialize filter
    push!(s̄.filter,(s̄.θ_max,Inf))

    if verbose
        println("θ0: $(s̄.θ), φ0: $(s̄.φ)")
        println("Eμ0: $(eval_Eμ(0.0,s̄))\n")
    end

    while eval_Eμ(0.0,s̄) > s̄.opts.ϵ_tol
        while eval_Eμ(s̄.μ,s̄) > s̄.opts.κϵ*s̄.μ
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
                    if s̄.θ < s̄.opts.ϵ_tol
                        @warn "infeasibility (restoration phase)"
                    else
                        restoration_reset!(s̄,s)
                    end
                else
                    augment_filter!(s̄)
                    update!(s̄)
                end
            end

            if check_filter(θ(s̄.x[s.idx.x],s),barrier(s̄.x[s.idx.x],s),s) && θ(s̄.x[s.idx.x],s) <= s̄.opts.κ_resto*s.θ
                println("-restoration phase: success\n")
                return true
            end

            s̄.opts.z_reset ? reset_z!(s̄) : nothing

            eval_iterate!(s̄)

            s̄.k += 1
            if s̄.k > s̄.opts.max_iter
                error("max iterations (restoration)")
            end

            if verbose
                println("restoration iteration ($(s̄.j),$(s̄.k)):")
                s.model.n < 5 ? println("x: $(s̄.x[s.idx.x])") : nothing
                println("θ: $(θ(s̄.x,s̄)), φ: $(barrier(s̄.x,s̄))")
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
    error("<phase 2 complete>: locally infeasible")
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

    f̄_func(x) = 0.

    function ∇f̄_func!(∇f,x)
        return nothing
    end
    function ∇²f̄_func!(∇²f,x)
        return nothing
    end

    function c̄_func!(c,x)
        return nothing
    end

    function ∇c̄_func!(∇c,x)
        return nothing
    end

    function ∇²c̄λ_func!(∇²c̄λ,x,λ)
        return nothing
    end

    _model = Model(n̄,m̄,x̄L,x̄U,f̄_func,∇f̄_func!,∇²f̄_func!,c̄_func!,∇c̄_func!,∇²c̄λ_func!)

    s̄ = Solver(x̄,_model,opts=opts)
    s̄.DR = spzeros(s.model.n,s.model.n)
    s̄.idx_r = restoration_indices(s)
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

    # # project
    # for i = 1:s̄.model.n
    #     s̄.x[i] = init_x0(s̄.x[i],s̄.xL[i],s̄.xU[i],s̄.opts.κ1,s̄.opts.κ2)
    # end

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

    s̄.model.∇f_func!(s̄.∇f,s̄.x)
    s̄.model.c_func!(s̄.c,s̄.x)
    s̄.model.∇c_func!(s̄.A,s̄.x)

    init_Dx!(s̄.Dx,s̄.model.n)
    s̄.df = init_df(s̄.opts.g_max,s̄.∇f)
    init_Dc!(s̄.Dc,s̄.opts.g_max,s̄.A,s̄.model.m)

    if s̄.opts.nlp_scaling
        s̄.c .= s̄.Dc*s̄.c
    end

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
        c .-= x[s̄.idx_r.p]
        c .+= x[s̄.idx_r.n]
        return nothing
    end

    function ∇c_func!(∇c,x)
        s.model.∇c_func!(view(∇c,1:s.model.m,s.idx.x),x[s.idx.x])
        ∇c[CartesianIndex.(1:s.model.m,s̄.idx_r.p)] .= -1.0
        ∇c[CartesianIndex.(1:s.model.m,s̄.idx_r.n)] .= 1.0
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
        search_direction_symmetric_restoration!(s̄,s)
    elseif s.opts.kkt_solve == :symmetric
        search_direction_symmetric_restoration!(s̄,s)
    else
        error("restoration does not have kkt solve method")
    end

    return small_search_direction(s̄)
end

function search_direction_unreduced_restoration!(s̄::Solver,s::Solver)
    kkt_hessian_symmetric!(s̄)
    inertia_correction!(s̄,restoration=s̄.restoration)

    kkt_hessian_unreduced!(s̄)
    kkt_gradient_unreduced!(s̄)

    s̄.d .= lu(s̄.H + Diagonal(s̄.δ))\(-s̄.h)

    s.opts.iterative_refinement ? iterative_refinement(s̄.d,s̄) : nothing

    return nothing
end
# symmetric KKT system
function kkt_hessian_symmetric_restoration!(s̄::Solver,s::Solver)
    s.model.∇²cλ_func!(s.W,s̄.x[s.idx.x],s̄.λ)
    s̄.ΣL[CartesianIndex.(s̄.idx.xL,s̄.idx.xL)] .= s̄.zL./((s̄.x - s̄.xL)[s̄.xL_bool])
    s̄.ΣU[CartesianIndex.(s̄.idx.xU,s̄.idx.xU)] .= s̄.zU./((s̄.xU - s̄.x)[s̄.xU_bool])
    s.ΣL .= s̄.ΣL[s.idx.x,s.idx.x]
    s.ΣU .= s̄.ΣU[s.idx.x,s.idx.x]

    s.model.∇c_func!(s.A,s̄.x[s.idx.x])

    p = s̄.x[s̄.idx_r.p]
    n = s̄.x[s̄.idx_r.n]

    zL = s̄.zL[1:s.nL]
    zp = s̄.zL[s.nL .+ (1:s.model.m)]
    zn = s̄.zL[s.nL + s.model.m .+ (1:s.model.m)]

    s.H_sym[s.idx.x,s.idx.x] .= s.∇²cλ + sqrt(s̄.μ)*s̄.DR'*s̄.DR + s.ΣL + s.ΣU
    s.H_sym[s.idx.x,s.idx.λ] .= s.A'
    s.H_sym[s.idx.λ,s.idx.x] .= s.A
    s.H_sym[s.idx.λ,s.idx.λ] .= -1.0*Diagonal(p./zp) - Diagonal(n./zn)

    return nothing
end

function kkt_gradient_symmetric_restoration!(s̄::Solver,s::Solver)
    s.model.c_func!(s.c,s̄.x[s.idx.x])
    s.model.∇c_func!(s.A,s̄.x[s.idx.x])

    p = s̄.x[s̄.idx_r.p]
    n = s̄.x[s̄.idx_r.n]

    λ = s̄.λ

    ρ = s̄.opts.ρ
    μ = s̄.μ

    zL = s̄.zL[1:s.nL]
    zp = s̄.zL[s.nL .+ (1:s.model.m)]
    zn = s̄.zL[s.nL + s.model.m .+ (1:s.model.m)]

    s.h_sym[s.idx.x] .= sqrt(μ)*s̄.DR'*s̄.DR*(s̄.x[s.idx.x] - s.x) + s.A'*s̄.λ
    s.h_sym[s.idx.xL] .-= μ./(s̄.x[s.idx.x] - s.xL)[s.xL_bool]
    s.h_sym[s.idx.xU] .+= μ./(s.xU - s̄.x[s.idx.x])[s.xU_bool]
    s.h_sym[s.idx.λ] .= s.c - p + n + ρ*Diagonal(zp)\(μ*ones(s.model.m) - p) + ρ*Diagonal(zn)\(μ*ones(s.model.m) - n)

    return nothing
end

function search_direction_symmetric_restoration!(s̄::Solver,s::Solver)
    kkt_hessian_symmetric_restoration!(s̄,s)
    kkt_gradient_symmetric_restoration!(s̄,s)

    inertia_correction!(s)

    s̄.d[s̄.idx_r.xλ] .= ma57_solve(s.LBL, -s.h_sym)
    dx = s̄.d[s.idx.x]
    dλ = s̄.d[s̄.idx.λ]

    x = s̄.x[s.idx.x]
    p = s̄.x[s̄.idx_r.p]
    n = s̄.x[s̄.idx_r.n]

    λ = s̄.λ

    zL = s̄.zL[1:s.nL]
    zp = s̄.zL[s.nL .+ (1:s.model.m)]
    zn = s̄.zL[s.nL + s.model.m .+ (1:s.model.m)]

    zU = s̄.zU[s̄.idx_r.zU]

    μ = s̄.μ
    ρ = s̄.opts.ρ

    Σp = Diagonal(zp./p)
    Σn = Diagonal(zn./n)

    # dp
    s̄.d[s̄.idx_r.p] .= Diagonal(zp)\(μ*ones(s.model.m) + Diagonal(p)*(λ + dλ) - ρ*p)
    dp = s̄.d[s̄.idx_r.p]

    # dn
    s̄.d[s̄.idx_r.n] .= Diagonal(zn)\(μ*ones(s.model.m) - Diagonal(n)*(λ + dλ) - ρ*n)
    dn = s̄.d[s̄.idx_r.n]

    # dzL
    zL_idx = (s̄.model.n + s.model.m) .+ (1:s.nL)
    s̄.d[zL_idx] .= -zL./((x - s.xL)[s.xL_bool]).*dx[s.xL_bool] - zL + μ./((x - s.xL)[s.xL_bool])
    dzL = s̄.d[zL_idx]

    #dzU
    zU_idx = (s̄.model.n + s.model.m + s.nL + s.model.m + s.model.m) .+ (1:s.nU)
    s̄.d[zU_idx] .= zU./((s.xU - x)[s.xU_bool]).*dx[s.xU_bool] - zU + μ./((s.xU - x)[s.xU_bool])
    dzU = s̄.d[zU_idx]

    # dzp
    zp_idx = (s̄.model.n + s.model.m + s.nL) .+ (1:s.model.m)
    s̄.d[zp_idx] .= μ*Diagonal(p)\ones(s.model.m) - zp - Σp*dp
    dzp = s̄.d[zp_idx]

    # dzn
    zn_idx = (s̄.model.n + s.model.m + s.nL + s.model.m) .+ (1:s.model.m)
    s̄.d[zn_idx] .= μ*Diagonal(n)\ones(s.model.m) - zn - Σn*dn
    dzn = s̄.d[zn_idx]

    s̄.opts.iterative_refinement ? iterative_refinement_restoration(s̄.d,s̄,s) : nothing

    return small_search_direction(s̄)
end

function iterative_refinement_restoration(d,s̄::Solver,s::Solver; verbose=true)
    s̄.d_copy = copy(d)
    iter = 0

    kkt_hessian_unreduced!(s̄)
    kkt_gradient_unreduced!(s̄)

    s̄.res = -s̄.h - s̄.H*d

    res_norm = norm(s̄.res,Inf)
    res_norm_init = copy(res_norm)

    verbose ? println("init res: $(res_norm), δw: $(s.δw), δc: $(s.δc)") : nothing

    x = s̄.x[s.idx.x]
    p = s̄.x[s̄.idx_r.p]
    n = s̄.x[s̄.idx_r.n]
    λ = s̄.λ
    zL = s̄.zL[1:s.nL]
    zp = s̄.zL[s.nL .+ (1:s.model.m)]
    zn = s̄.zL[s.nL + s.model.m .+ (1:s.model.m)]
    zU = s̄.zU[1:s.nU]

    xL = (x - s.xL)[s.xL_bool]
    xU = (s.xU - x)[s.xU_bool]

    while (iter < s̄.opts.max_iterative_refinement && res_norm > s̄.opts.ϵ_iterative_refinement) || iter < s̄.opts.min_iterative_refinement

        r1 = s̄.res[s.idx.x]
        r2 = s̄.res[s̄.idx_r.p]
        r3 = s̄.res[s̄.idx_r.n]
        r4 = s̄.res[s̄.idx.λ]
        r5 = s̄.res[s̄.model.n + s.model.m .+ (1:s.nL)]
        r6 = s̄.res[s̄.model.n + s.model.m + s.nL .+ (1:s.model.m)]
        r7 = s̄.res[s̄.model.n + s.model.m + s.nL + s.model.m .+ (1:s.model.m)]
        r8 = s̄.res[s̄.model.n + s.model.m + s.nL + s.model.m + s.model.m .+ (1:s.nU)]

        s̄.res[s.idx.xL] .+= r5./xL
        s̄.res[s.idx.xU] .-= r8./xU
        s̄.res[s̄.model.n .+ (1:s.model.m)] .+= p./zp.*r2 + r6./zp - n./zn.*r3 - r7./zn

        s̄.Δ[s̄.idx_r.xλ] .= ma57_solve(s.LBL,s̄.res[s̄.idx_r.xλ])

        dx = s̄.Δ[s.idx.x]
        dλ = s̄.Δ[s̄.idx.λ]

        s̄.Δ[s̄.idx_r.p] .= -p.*(-dλ - r2)./zp + r6./zp
        s̄.Δ[s̄.idx_r.n] .= -n.*(dλ - r3)./zn + r7./zn
        s̄.Δ[s.model.n + 3s.model.m .+ (1:s.nL)] .= -zL./xL.*dx[s.xL_bool] + r5./xL
        s̄.Δ[s.model.n + 3s.model.m + s.nL .+ (1:s.model.m)] .= -dλ - r2
        s̄.Δ[s.model.n + 4s.model.m + s.nL .+ (1:s.model.m)] .= dλ - r3
        s̄.Δ[s.model.n + 5s.model.m + s.nL .+ (1:s.nU)] .= zU./xU.*dx[s.xU_bool] + r8./xU

        d .+= s̄.Δ
        s̄.res = -s̄.h - s̄.H*d

        res_norm = norm(s̄.res,Inf)

        iter += 1
    end

    if res_norm < s̄.opts.ϵ_iterative_refinement# || res_norm < res_norm_init
        verbose ? println("iterative refinement success: $(res_norm), iter: $iter, cond: $(cond(Array(s̄.H+Diagonal(s̄.δ)))), rank: $(rank(Array(s̄.H+Diagonal(s̄.δ))))") : nothing
        return true
    else
        d .= s̄.d_copy
        verbose ? println("iterative refinement failure: $(res_norm), iter: $iter, cond: $(cond(Array(s̄.H+Diagonal(s̄.δ)))), rank: $(rank(Array(s̄.H+Diagonal(s̄.δ))))") : nothing
        return false
    end
end

function restoration_indices(s::Solver)
    p = s.model.n .+ (1:s.model.m)
    n = s.model.n + s.model.m .+ (1:s.model.m)
    zL = 1:s.nL
    zp = s.nL .+ (1:s.model.m)
    zn = s.nL + s.model.m .+ (1:s.model.m)
    zU = 1:s.nU
    xλ = [s.idx.x...,(s.model.n + s.model.m + s.model.m .+ (1:s.model.m))...]

    RestorationIndices(p,n,zL,zp,zn,zU,xλ)
end
