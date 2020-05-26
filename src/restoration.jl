function restoration!(s̄::Solver,s::Solver)
    s̄.opts.verbose ? println("~restoration phase~\n") : nothing
    if !kkt_error_reduction(s)
        # phase 2 solver
        initialize_restoration_solver!(s̄,s)

        # solve phase 2
        solve_restoration!(s̄,s,verbose=true)

        # update phase 1 solver
        update_phase1_solver!(s̄,s)
    else
        s̄.opts.verbose ? println("-KKT error reduction: success") : nothing
    end
    return nothing
end

function update_phase1_solver!(s̄::Solver,s::Solver)
    s.dx .= view(s̄.x,s.idx.x) - s.x
    s.dzL .= -s.σL.*s.dxL - s.zL + s.μ./s.ΔxL
    s.dzU .= s.σU.*s.dxU - s.zU + s.μ./s.ΔxU

    αz_max!(s)

    s.x .= view(s̄.x,s.idx.x)

    # project phase 2 solution on phase 1 bounds
    # for i = 1:s.n
    #     s.x[i] = init_x0(s̄.x[i],s.xL[i],s.xU[i],s.opts.κ1,s.opts.κ2)
    # end

    s.zL .+= s.αz*s.dzL
    s.zU .+= s.αz*s.dzU

    eval_∇f!(s.model,s.x)
    eval_∇c!(s.model,s.x)
    init_y!(s.y,s.H_sym,s.h_sym,s.d,s.zL,s.zU,get_∇f(s.model),get_∇c(s.model),s.model.n,s.model.m,s.model.xL_bool,s.model.xU_bool,s.opts.y_max)

    return nothing
end

function solve_restoration!(s̄::Solver,s::Solver; verbose=false)
    # evaluate problem
    s̄.opts.verbose = true
    eval_step!(s̄)

    # initialize filter
    push!(s̄.filter,(s̄.θ_max,Inf))

    if s̄.opts.verbose
        println("   θ0: $(s̄.θ), φ0: $(s̄.φ)")
        println("   Eμ0: $(eval_Eμ(0.0,s̄))\n")
    end

    while eval_Eμ(0.0,s̄) > s̄.opts.ϵ_tol
        while eval_Eμ(s̄.μ,s̄) > s̄.opts.κϵ*s̄.μ
            if search_direction_restoration!(s̄,s)
                s̄.small_search_direction_cnt += 1
                if s̄.small_search_direction_cnt == s̄.opts.small_search_direction_max
                    s̄.small_search_direction_cnt = 0
                    if s̄.μ < 0.1*s̄.opts.ϵ_tol
                        s̄.opts.verbose ? println("<phase 2 complete>: small search direction") : nothing
                        return
                    end
                else
                    break
                end
                α_max!(s̄)
                αz_max!(s̄)
                augment_filter!(s̄)
                accept_step!(s̄)
            else
                s̄.small_search_direction_cnt = 0

                if !line_search(s̄)
                    if s̄.θ < s̄.opts.ϵ_tol
                        @warn "infeasibility (restoration phase)"
                        return
                    else
                        restoration_reset!(s̄,s)
                    end
                else
                    augment_filter!(s̄)
                    accept_step!(s̄)
                end
            end

            if check_filter(θ(view(s̄.x,s.idx.x),s),barrier(view(s̄.x,s.idx.x),s),s.filter) && θ(view(s̄.x,s.idx.x),s) <= s̄.opts.κ_resto*s.θ
                s̄.opts.verbose ? println("-restoration phase: success\n") : nothing
                return true
            end

            s̄.opts.z_reset && reset_z!(s̄)

            eval_step!(s̄)

            s̄.k += 1
            if s̄.k > s̄.opts.max_iter
                error("max iterations (restoration)")
            end

            if s̄.opts.verbose
                println("   restoration iteration ($(s̄.j),$(s̄.k)):")
                s.model.n < 5 ? println("   x: $(s̄.x[s.idx.x])") : nothing
                println("   θ: $(θ(s̄.x,s̄)), φ: $(barrier(s̄.x,s̄))")
                println("   Eμ: $(eval_Eμ(s̄.μ,s̄))")
                println("   α: $(s̄.α)\n")
            end
        end

        barrier_update!(s̄)
        augmented_lagrangian_update!(s̄)
        eval_step!(s̄)

        if s̄.k == 0
            barrier_update!(s̄)
            augmented_lagrangian_update!(s̄)
            eval_step!(s̄)
        end

        update_restoration_model_info!(s̄)
    end
    @warn "<phase 2 complete>: locally infeasible"
    return
end

function restoration_reset!(s̄::Solver,s::Solver)
    eval_c!(s.model,view(s̄.x,s.idx.x))
    get_c_scaled!(s.c,s)

    # initialize p,n
    n = s.model.n
    m = s.model.m
    for i = 1:m
        s̄.x[n + m + i] = init_n(s.c[i],s̄.μ,s̄.opts.ρ_resto)
    end

    for i = 1:m
        s̄.x[n + i] = init_p(s̄.x[n + m + i],s.c[i])
    end
    s̄.y .= 0

    return nothing
end

function RestorationSolver(s::Solver)
    opts_r = copy(s.opts)
    opts_r.y_init_ls = false
    opts_r.relax_bnds = false
    opts_r.kkt_solve = :symmetric
    model_r = restoration_model(s.model,bnd_tol=s.opts.bnd_tol)

    s̄ = Solver(zeros(model_r.n),model_r,s.model_opt,opts=opts_r)
    s̄.idx_r = restoration_indices(s̄,s)
    return s̄
end

function initialize_restoration_solver!(s̄::Solver,s::Solver)
    s̄.k = 0
    s̄.j = 0

    eval_c!(s.model,view(s̄.x,s.idx.x))
    get_c_scaled!(s.c,s)

    s̄.μ = max(s.μ,norm(s.c,Inf))
    s̄.τ = update_τ(s̄.μ,s̄.opts.τ_min)

    s̄.ρ = 1/s̄.μ
    s̄.λ .= 0.

    s̄.x[s.idx.x] = copy(s.x)

    # initialize p,n
    for i = 1:s.model.m
        s̄.x[s.model.n + s.model.m + i] = init_n(s.c[i],s̄.μ,s̄.opts.ρ_resto)
    end

    for i = 1:s.model.m
        s̄.x[s.model.n + i] = init_p(s̄.x[s.model.n + s.model.m + i],s.c[i])
    end

    # # project
    # for i = 1:s̄.n
    #     s̄.x[i] = init_x0(s̄.x[i],s̄.xL[i],s̄.xU[i],s̄.opts.κ1,s̄.opts.κ2)
    # end

    # initialize zL, zU, zp, zn
    for i = 1:s.model.nL
        s̄.zL[i] = min(s̄.opts.ρ_resto,s.zL[i])
    end

    for i = 1:s.model.nU
        s̄.zU[i] = min(s̄.opts.ρ_resto,s.zU[i])
    end

    s̄.zL[s.model.nL .+ (1:2s.model.m)] = s̄.μ./view(s̄.x,s.model.n .+ (1:2s.model.m))

    s̄.restoration = true

    init_restoration_model_info!(s̄,s)

    empty!(s̄.filter)

    eval_∇f!(s̄.model,s̄.x)

    eval_∇c!(s̄.model,s̄.x)
    s̄.df = init_df(s̄.opts.g_max,get_∇f(s̄.model))
    init_Dc!(s̄.Dc,s̄.opts.g_max,get_∇c(s̄.model),s̄.model.m)

    eval_c!(s̄.model,s̄.x)
    get_c_scaled!(s̄.c,s̄)


    s̄.θ = norm(s̄.c,1)
    s̄.θ_min = init_θ_min(s̄.θ)
    s̄.θ_max = init_θ_max(s̄.θ)

    return nothing
end

function init_restoration_model_info!(s̄::Solver,s::Solver)
    s̄.model.info.xR = s.x
    init_DR!(s̄.model.info.DR,s.x,s.model.n)
    s̄.model.info.ζ = sqrt(s̄.μ)
    s̄.model.info.ρ = s̄.opts.ρ_resto
    return nothing
end

function update_restoration_model_info!(s̄::Solver)
    s̄.model.info.ζ = sqrt(s̄.μ)
    return nothing
end

function init_DR!(DR,xr,n)
    for i = 1:n
        DR[i,i] = min(1.0,1.0/abs(xr[i]))
    end
    return nothing
end

function init_n(c,μ,ρ_resto)
    n = (μ - ρ_resto*c)/(2.0*ρ_resto) + sqrt(((μ-ρ_resto*c)/(2.0*ρ_resto))^2 + (μ*c)/(2.0*ρ_resto))
    return n
end

function init_p(n,c)
    p = c + n
    return p
end

function search_direction_restoration!(s̄::Solver,s::Solver)
    if s.opts.kkt_solve == :fullspace
        search_direction_symmetric_restoration!(s̄,s)
    elseif s.opts.kkt_solve == :symmetric
        search_direction_symmetric_restoration!(s̄,s)
    else
        error("restoration does not have kkt solve method")
    end

    return small_search_direction(s̄)
end

function search_direction_fullspace_restoration!(s̄::Solver,s::Solver)
    kkt_hessian_symmetric!(s̄)
    inertia_correction!(s̄,restoration=s̄.restoration)

    kkt_hessian_fullspace!(s̄)

    s̄.d .= lu(s̄.H + Diagonal(s̄.δ))\(-s̄.h)

    s.opts.iterative_refinement ? iterative_refinement(s̄.d,s̄) : nothing

    return nothing
end

# symmetric KKT system
function kkt_hessian_symmetric_restoration!(s̄::Solver,s::Solver)
    update!(s.Hv_sym.xx,s̄.∇²L[s.idx.x,s.idx.x])
    add_update!(s.Hv_sym.xLxL,view(s̄.σL,s̄.idx_r.zLL))
    add_update!(s.Hv_sym.xUxU,view(s̄.σU,s̄.idx_r.zUU))
    s.Hv_sym.xy .= view(get_∇c(s̄.model),1:s.model.m,s.idx.x)'
    s.Hv_sym.yx .= view(get_∇c(s̄.model),1:s.model.m,s.idx.x)
    update!(s.Hv_sym.yy,-1.0./view(s̄.σL,s̄.idx_r.zLp) - 1.0./view(s̄.σL,s̄.idx_r.zLn))
    return nothing
end

function kkt_gradient_symmetric_restoration!(s̄::Solver,s::Solver)
    s.h_sym[s.idx.x] = s̄.h[s.idx.x]
    s.h_sym[s.idx.xL] += (view(s̄.h,s̄.idx.zL)./s̄.ΔxL)[s̄.idx_r.zLL]
    s.h_sym[s.idx.xU] -= (view(s̄.h,s̄.idx.zU)./s̄.ΔxU)[s̄.idx_r.zUU]
    s.h_sym[s.idx.y] = view(s̄.h,s̄.idx.y) + 1.0./view(s̄.σL,s̄.idx_r.zLp).*view(s̄.h,s̄.idx_r.p) -1.0./view(s̄.σL,s̄.idx_r.zLn).*view(s̄.h,s̄.idx_r.n) + view(s̄.h,s̄.idx_r.zp)./view(s̄.zL,s̄.idx_r.zLp) - view(s̄.h,s̄.idx_r.zn)./view(s̄.zL,s̄.idx_r.zLn)
    return nothing
end

function search_direction_symmetric_restoration!(s̄::Solver,s::Solver)
    kkt_hessian_symmetric_restoration!(s̄,s)
    kkt_gradient_symmetric_restoration!(s̄,s)

    inertia_correction!(s)

    rx = view(-s̄.h,s.idx.x)
    rp = view(-s̄.h,s̄.idx_r.p)
    rn = view(-s̄.h,s̄.idx_r.n)
    ry = view(-s̄.h,s̄.idx.y)
    rzL = view(-s̄.h,s̄.idx_r.zL)
    rzp = view(-s̄.h,s̄.idx_r.zp)
    rzn = view(-s̄.h,s̄.idx_r.zn)
    rzU = view(-s̄.h,s̄.idx_r.zU)

    s̄.d[s̄.idx_r.xy] = ma57_solve(s.LBL,-s.h_sym)

    dx = view(s̄.d,s.idx.x)
    dy = view(s̄.d,s̄.idx.y)

    s̄.d[s̄.idx_r.p] = -1.0./view(s̄.σL,s̄.idx_r.zLp).*(-dy - rp) + rzp./view(s̄.zL,s̄.idx_r.zLp)
    s̄.d[s̄.idx_r.n] = -1.0./view(s̄.σL,s̄.idx_r.zLn).*(dy - rn) + rzn./view(s̄.zL,s̄.idx_r.zLn)
    s̄.d[s̄.idx_r.zL] = -view(s̄.σL,s̄.idx_r.zLL).*view(dx,s.idx.xL) + rzL./view(s̄.ΔxL,s̄.idx_r.zLL)
    s̄.d[s̄.idx_r.zp] = -dy - rp
    s̄.d[s̄.idx_r.zn] = dy - rn
    s̄.d[s̄.idx_r.zU] = view(s̄.σU,s̄.idx_r.zUU).*view(dx,s.idx.xU) + rzU./view(s̄.ΔxU,s̄.idx_r.zUU)

    if s̄.opts.iterative_refinement
        kkt_hessian_fullspace!(s̄)
        iterative_refinement_restoration(s̄.d,s̄,s)
    end

    return small_search_direction(s̄)
end

function iterative_refinement_restoration(d::Vector{T},s̄::Solver,s::Solver) where T
    s̄.d_copy .= d
    iter = 0

    s̄.res .= -s̄.h - s̄.H*d

    res_norm = norm(s̄.res,Inf)

    s̄.opts.verbose ? println("init res: $(res_norm), δw: $(s.δw), δc: $(s.δc)") : nothing

    while (iter < s̄.opts.max_iterative_refinement && res_norm > s̄.opts.ϵ_iterative_refinement) || iter < s̄.opts.min_iterative_refinement

        if s̄.opts.kkt_solve == :fullspace
            s̄.Δ .= (s̄.H+Diagonal(s̄.δ))\s̄.res
        elseif s̄.opts.kkt_solve == :symmetric
            rx = view(s̄.res,s.idx.x)
            rp = view(s̄.res,s̄.idx_r.p)
            rn = view(s̄.res,s̄.idx_r.n)
            ry = view(s̄.res,s̄.idx.y)
            rzL = view(s̄.res,s̄.idx_r.zL)
            rzp = view(s̄.res,s̄.idx_r.zp)
            rzn = view(s̄.res,s̄.idx_r.zn)
            rzU = view(s̄.res,s̄.idx_r.zU)

            s.h_sym[s.idx.x] = rx
            s.h_sym[s.idx.xL] += rzL./view(s̄.ΔxL,s̄.idx_r.zLL)
            s.h_sym[s.idx.xU] -= rzU./view(s̄.ΔxU,s̄.idx_r.zUU)
            s.h_sym[s.idx.y] = ry
            s.h_sym[s.idx.y] += 1.0./view(s̄.σL,s̄.idx_r.zLp).*rp + rzp./view(s̄.zL,s̄.idx_r.zLp) - 1.0./view(s̄.σL,s̄.idx_r.zLn).*rn - rzn./view(s̄.zL,s̄.idx_r.zLn)

            s̄.Δ[s̄.idx_r.xy] = ma57_solve(s.LBL,s.h_sym)

            dx = view(s̄.Δ,s.idx.x)
            dy = view(s̄.Δ,s̄.idx.y)

            s̄.Δ[s̄.idx_r.p] = -1.0./view(s̄.σL,s̄.idx_r.zLp).*(-dy - rp) + rzp./view(s̄.zL,s̄.idx_r.zLp)
            s̄.Δ[s̄.idx_r.n] = -1.0./view(s̄.σL,s̄.idx_r.zLn).*(dy - rn) + rzn./view(s̄.zL,s̄.idx_r.zLn)
            s̄.Δ[s̄.idx_r.zL] = -view(s̄.σL,s̄.idx_r.zLL).*view(dx,s.idx.xL) + rzL./view(s̄.ΔxL,s̄.idx_r.zLL)
            s̄.Δ[s̄.idx_r.zp] = -dy - rp
            s̄.Δ[s̄.idx_r.zn] = dy - rn
            s̄.Δ[s̄.idx_r.zU] = view(s̄.σU,s̄.idx_r.zUU).*view(dx,s.idx.xU) + rzU./view(s̄.ΔxU,s̄.idx_r.zUU)
        end

        d .+= s̄.Δ
        s̄.res .= -s̄.h - s̄.H*d

        res_norm = norm(s̄.res,Inf)

        iter += 1
    end

    if res_norm < s̄.opts.ϵ_iterative_refinement# || res_norm < res_norm_init
        println("iterative refinement success: $(res_norm), iter: $iter")# : nothing#, cond: $(cond(Array(s̄.H+Diagonal(s̄.δ)))), rank: $(rank(Array(s̄.H+Diagonal(s̄.δ))))") : nothing
        return true
    else
        d .= s̄.d_copy
        println("iterative refinement failure: $(res_norm), iter: $iter")# : nothing#, cond: $(cond(Array(s̄.H+Diagonal(s̄.δ)))), rank: $(rank(Array(s̄.H+Diagonal(s̄.δ))))") : nothing
        return false
    end
end

function restoration_indices(s̄::Solver,s::Solver)
    p = s.model.n .+ (1:s.model.m)
    n = s.model.n + s.model.m .+ (1:s.model.m)
    zL = s̄.model.n + s̄.model.m .+ (1:s.model.nL)
    zp = s̄.model.n + s̄.model.m + s.model.nL .+ (1:s.model.m)
    zn = s̄.model.n + s̄.model.m + s.model.nL + s.model.m .+ (1:s.model.m)
    zU = s̄.model.n + s̄.model.m + s.model.nL + s.model.m + s.model.m .+ (1:s.model.nU)
    xy = [s.idx.x...,(s̄.model.n .+ (1:s̄.model.m))...]

    zLL = 1:s.model.nL
    zLp = s.model.nL .+ (1:s.model.m)
    zLn = s.model.nL + s.model.m .+ (1:s.model.m)
    zUU = 1:s.model.nU

    RestorationIndices(p,n,zL,zp,zn,zU,xy,zLL,zLp,zLn,zUU)
end

function restoration_indices(model_r::Model,model::Model)
    p = model.n .+ (1:model.m)
    n = model.n + model.m .+ (1:model.m)
    zL = model_r.n + model_r.m .+ (1:model.nL)
    zp = model_r.n + model_r.m + model.nL .+ (1:model.m)
    zn = model_r.n + model_r.m + model.nL + model.m .+ (1:model.m)
    zU = model_r.n + model_r.m + model.nL + model.m + model.m .+ (1:model.nU)
    xy = [(1:model.n)...,(model_r.n .+ (1:model_r.m))...]

    zLL = 1:model.nL
    zLp = model.nL .+ (1:model.m)
    zLn = model.nL + model.m .+ (1:model.m)
    zUU = 1:model.nU

    RestorationIndices(p,n,zL,zp,zn,zU,xy,zLL,zLp,zLn,zUU)
end
