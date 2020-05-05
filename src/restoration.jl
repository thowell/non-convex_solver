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

    s.∇f_func!(s.∇f,s.x,s.model)
    s.∇c_func!(s.∇c,s.x,s.model)
    init_y!(s.y,s.H_sym,s.h_sym,s.d,s.zL,s.zU,s.∇f,s.∇c,s.n,s.m,s.xL_bool,s.xU_bool,s.opts.y_max)

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
                s.n < 5 ? println("   x: $(s̄.x[s.idx.x])") : nothing
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

        update_restoration_objective!(s̄,s)
    end
    @warn "<phase 2 complete>: locally infeasible"
    return
end

function restoration_reset!(s̄::Solver,s::Solver)
    s.c_func!(s.c,view(s̄.x,s.idx.x),s.model)
    s.cA .+= 1.0/s̄.ρ*(s̄.λ - s̄.yA)

    # initialize p,n
    for i = 1:s.m
        s̄.x[s.n + s.m + i] = init_n(s.c[i],s̄.μ,s̄.opts.ρ_resto)
    end

    for i = 1:s.m
        s̄.x[s.n + i] = init_p(s̄.x[s.n + s.m + i],s.c[i])
    end
    s̄.y .= 0

    return nothing
end

function RestorationSolver(s::Solver)
    opts = copy(s.opts)
    opts.y_init_ls = false
    opts.relax_bnds = false

    n̄ = s.n + 2s.m
    m̄ = s.m

    x̄ = zeros(n̄)

    x̄L = zeros(n̄)
    x̄L[s.idx.x] = s.xL # maybe initialized with phase 1 relaxed bounds

    x̄U = Inf*ones(n̄)
    x̄U[s.idx.x] = s.xU # maybe initialize with phase 1 relaxed bounds

    f̄_func(x,model::AbstractModel) = 0.

    function ∇f̄_func!(∇f,x,model::AbstractModel)
        return nothing
    end
    function ∇²f̄_func!(∇²f,x,model::AbstractModel)
        return nothing
    end

    function c̄_func!(c,x,model::AbstractModel)
        return nothing
    end

    function ∇c̄_func!(∇c,x,model::AbstractModel)
        return nothing
    end

    function ∇²c̄y_func!(∇²c̄y,x,y,model::AbstractModel)
        return nothing
    end

    _model = Model(n̄,m̄,x̄L,x̄U,f̄_func,∇f̄_func!,∇²f̄_func!,c̄_func!,∇c̄_func!,∇²c̄y_func!)

    s̄ = Solver(x̄,_model,cA_idx=s.cA_idx,opts=opts)
    s̄.DR = spzeros(s.n,s.n)
    s̄.idx_r = restoration_indices(s)
    return s̄
end

function initialize_restoration_solver!(s̄::Solver,s::Solver)
    s̄.k = 0
    s̄.j = 0

    s.c_func!(s.c,view(s̄.x,s.idx.x),s.model)

    s̄.μ = max(s.μ,norm(s.c,Inf))
    s̄.τ = update_τ(s̄.μ,s̄.opts.τ_min)

    s̄.ρ = 1/s̄.μ
    s̄.λ .= 0.

    s̄.x[s.idx.x] = copy(s.x)

    # initialize p,n
    for i = 1:s.m
        s̄.x[s.n + s.m + i] = init_n(s.c[i],s̄.μ,s̄.opts.ρ_resto)
    end

    for i = 1:s.m
        s̄.x[s.n + i] = init_p(s̄.x[s.n + s.m + i],s.c[i])
    end

    # # project
    # for i = 1:s̄.n
    #     s̄.x[i] = init_x0(s̄.x[i],s̄.xL[i],s̄.xU[i],s̄.opts.κ1,s̄.opts.κ2)
    # end

    # initialize zL, zU, zp, zn
    for i = 1:s.nL
        s̄.zL[i] = min(s̄.opts.ρ_resto,s.zL[i])
    end

    for i = 1:s.nU
        s̄.zU[i] = min(s̄.opts.ρ_resto,s.zU[i])
    end

    s̄.zL[s.nL .+ (1:2s.m)] = s̄.μ./view(s̄.x,s.n .+ (1:2s.m))

    init_DR!(s̄.DR,s.x,s.n)

    s̄.restoration = true

    update_restoration_objective!(s̄,s)
    update_restoration_constraints!(s̄,s)
    empty!(s̄.filter)

    s̄.∇f_func!(s̄.∇f,s̄.x,s̄.model)
    s̄.c_func!(s̄.c,s̄.x,s̄.model)
    s̄.∇c_func!(s̄.∇c,s̄.x,s̄.model)

    init_Dx!(s̄.Dx,s̄.n)
    s̄.df = init_df(s̄.opts.g_max,s̄.∇f)
    init_Dc!(s̄.Dc,s̄.opts.g_max,s̄.∇c,s̄.m)

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
    idx_pn = s.n .+ (1:2s.m)

    function f_func(x,model::AbstractModel)
        s̄.opts.ρ_resto*sum(view(x,idx_pn)) + 0.5*ζ*(view(x,s.idx.x) - s.x)'*DR'*DR*(view(x,s.idx.x) - s.x)
    end

    function ∇f_func!(∇f,x,model::AbstractModel)
        ∇f[s.idx.x] = ζ*DR'*DR*(view(x,s.idx.x) - s.x)
        ∇f[idx_pn] .= s̄.opts.ρ_resto
        return nothing
    end

    function ∇²f_func!(∇²f,x,model::AbstractModel)
        ∇²f[s.idx.x,s.idx.x] .= ζ*DR'*DR
        return nothing
    end

    s̄.model.f_func = f_func
    s̄.model.∇f_func! = ∇f_func!
    s̄.model.∇²f_func! = ∇²f_func!

    return nothing
end

function update_restoration_constraints!(s̄::Solver,s::Solver)
    function c_func!(c,x,model::AbstractModel)
        s.c_func!(c,x[s.idx.x],s.model)
        c .-= view(x,s̄.idx_r.p)
        c .+= view(x,s̄.idx_r.n)
        return nothing
    end

    function ∇c_func!(∇c,x,model::AbstractModel)
        s.∇c_func!(view(∇c,1:s.m,s.idx.x),x[s.idx.x],s.model)
        ∇c[CartesianIndex.(1:s.m,s̄.idx_r.p)] .= -1.0
        ∇c[CartesianIndex.(1:s.m,s̄.idx_r.n)] .= 1.0
        return nothing
    end

    function ∇²cy_func!(∇²cy,x,y,model::AbstractModel)
        s.∇²cy_func!(view(∇²cy,s.idx.x,s.idx.x),x[s.idx.x],y,s.model)
        return return nothing
    end

    s̄.c_func! = c_func!
    s̄.∇c_func! = ∇c_func!
    s̄.∇²cy_func! = ∇²cy_func!

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
    s.∇²cy_func!(s.∇²cy,s̄.x[s.idx.x],s̄.y,s.model)
    s.∇c_func!(s.∇c,s̄.x[s.idx.x],s.model)

    p = view(s̄.x,s̄.idx_r.p)
    n = view(s̄.x,s̄.idx_r.n)

    zL = view(s̄.zL,1:s.nL)
    zp = view(s̄.zL,s.nL .+ (1:s.m))
    zn = view(s̄.zL,s.nL + s.m .+ (1:s.m))

    update!(s.Hv_sym.xx,s.∇²cy + sqrt(s̄.μ)*s̄.DR'*s̄.DR)
    add_update!(s.Hv_sym.xLxL,view(s̄.σL,1:s.nL))
    add_update!(s.Hv_sym.xUxU,view(s̄.σU,1:s.nU))
    update!(s.Hv_sym.xy,s.∇c')
    update!(s.Hv_sym.yx,s.∇c)
    update!(s.Hv_sym.yy,-1.0*p./zp - n./zn)
    add_update!(s.Hv_sym.yalyal,-1.0/s̄.ρ)

    return nothing
end

function kkt_gradient_symmetric_restoration!(s̄::Solver,s::Solver)
    s.c_func!(s.c,view(s̄.x,s.idx.x),s.model)
    s.∇c_func!(s.∇c,view(s̄.x,s.idx.x),s.model)

    p = view(s̄.x,s̄.idx_r.p)
    n = view(s̄.x,s̄.idx_r.n)

    y = s̄.y

    ρ_resto = s̄.opts.ρ_resto
    μ = s̄.μ

    zL = view(s̄.zL,1:s.nL)
    zp = view(s̄.zL,s.nL .+ (1:s.m))
    zn = view(s̄.zL,s.nL + s.m .+ (1:s.m))

    # TODO damping

    s.h_sym[s.idx.x] = sqrt(μ)*s̄.DR'*s̄.DR*(view(s̄.x,s.idx.x) - s.x) + s.∇c'*s̄.y
    s.h_sym[s.idx.xL] -= μ./(view(s̄.x,s.idx.x) - s.xL)[s.xL_bool]
    s.h_sym[s.idx.xU] += μ./(s.xU - view(s̄.x,s.idx.x))[s.xU_bool]
    s.h_sym[s.idx.y] = s.c - p + n + ρ_resto*Diagonal(zp)\(μ .- p) + ρ_resto*Diagonal(zn)\(μ .- n)
    s.h_sym[s.idx.yA] += 1.0/s̄.ρ*(s̄.λ - s̄.yA)
    return nothing
end

function search_direction_symmetric_restoration!(s̄::Solver,s::Solver)
    kkt_hessian_symmetric_restoration!(s̄,s)
    kkt_gradient_symmetric_restoration!(s̄,s)

    inertia_correction!(s)

    s̄.d[s̄.idx_r.xy] = ma57_solve(s.LBL, -s.h_sym)
    dx = view(s̄.d,s.idx.x)
    dy = view(s̄.d,s̄.idx.y)

    x = view(s̄.x,s.idx.x)
    p = view(s̄.x,s̄.idx_r.p)
    n = view(s̄.x,s̄.idx_r.n)

    y = s̄.y

    zL = view(s̄.zL,1:s.nL)
    zp = view(s̄.zL,s.nL .+ (1:s.m))
    zn = view(s̄.zL,s.nL + s.m .+ (1:s.m))

    zU = view(s̄.zU,s̄.idx_r.zU)

    μ = s̄.μ
    ρ_resto = s̄.opts.ρ_resto

    #TODO grab from σL, σU
    Σp = Diagonal(zp./p)
    Σn = Diagonal(zn./n)

    # dp
    s̄.d[s̄.idx_r.p] = Diagonal(zp)\(μ .+ Diagonal(p)*(y + dy) - ρ_resto*p)
    dp = s̄.d[s̄.idx_r.p]

    # dn
    s̄.d[s̄.idx_r.n] = Diagonal(zn)\(μ .- Diagonal(n)*(y + dy) - ρ_resto*n)
    dn = view(s̄.d,s̄.idx_r.n)

    # dzL
    zL_idx = (s̄.n + s.m) .+ (1:s.nL)
    s̄.d[zL_idx] = -zL./((x - s.xL)[s.xL_bool]).*dx[s.xL_bool] - zL + μ./((x - s.xL)[s.xL_bool])
    dzL = view(s̄.d,zL_idx)

    #dzU
    zU_idx = (s̄.n + s.m + s.nL + s.m + s.m) .+ (1:s.nU)
    s̄.d[zU_idx] = zU./((s.xU - x)[s.xU_bool]).*dx[s.xU_bool] - zU + μ./((s.xU - x)[s.xU_bool])
    dzU = view(s̄.d,zU_idx)

    # dzp
    zp_idx = (s̄.n + s.m + s.nL) .+ (1:s.m)
    s̄.d[zp_idx] = μ*Diagonal(p)\ones(s.m) - zp - Σp*dp
    dzp = view(s̄.d,zp_idx)

    # dzn
    zn_idx = (s̄.n + s.m + s.nL + s.m) .+ (1:s.m)
    s̄.d[zn_idx] = μ*Diagonal(n)\ones(s.m) - zn - Σn*dn
    dzn = view(s̄.d,zn_idx)

    if s̄.opts.iterative_refinement
        kkt_hessian_unreduced!(s̄)
        kkt_gradient_unreduced!(s̄)
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

    x = view(s̄.x,s.idx.x)
    p = view(s̄.x,s̄.idx_r.p)
    n = view(s̄.x,s̄.idx_r.n)
    y = s̄.y
    zL = view(s̄.zL,1:s.nL)
    zp = view(s̄.zL,s.nL .+ (1:s.m))
    zn = view(s̄.zL,s.nL + s.m .+ (1:s.m))
    zU = view(s̄.zU,1:s.nU)

    xL = (x - s.xL)[s.xL_bool]
    xU = (s.xU - x)[s.xU_bool]

    while (iter < s̄.opts.max_iterative_refinement && res_norm > s̄.opts.ϵ_iterative_refinement) || iter < s̄.opts.min_iterative_refinement

        r1 = view(s̄.res,s.idx.x)
        r2 = view(s̄.res,s̄.idx_r.p)
        r3 = view(s̄.res,s̄.idx_r.n)
        r4 = view(s̄.res,s̄.idx.y)
        r5 = view(s̄.res,s̄.n + s.m .+ (1:s.nL))
        r6 = view(s̄.res,s̄.n + s.m + s.nL .+ (1:s.m))
        r7 = view(s̄.res,s̄.n + s.m + s.nL + s.m .+ (1:s.m))
        r8 = view(s̄.res,s̄.n + s.m + s.nL + s.m + s.m .+ (1:s.nU))

        s̄.res[s.idx.xL] += r5./xL
        s̄.res[s.idx.xU] -= r8./xU
        s̄.res[s̄.n .+ (1:s.m)] += p./zp.*r2 + r6./zp - n./zn.*r3 - r7./zn

        s̄.Δ[s̄.idx_r.xy] = ma57_solve(s.LBL,s̄.res[s̄.idx_r.xy])

        dx = view(s̄.Δ,s.idx.x)
        dy = view(s̄.Δ,s̄.idx.y)

        s̄.Δ[s̄.idx_r.p] = -p.*(-dy - r2)./zp + r6./zp
        s̄.Δ[s̄.idx_r.n] = -n.*(dy - r3)./zn + r7./zn
        s̄.Δ[s.n + 3s.m .+ (1:s.nL)] = -zL./xL.*dx[s.xL_bool] + r5./xL
        s̄.Δ[s.n + 3s.m + s.nL .+ (1:s.m)] = -dy - r2
        s̄.Δ[s.n + 4s.m + s.nL .+ (1:s.m)] = dy - r3
        s̄.Δ[s.n + 5s.m + s.nL .+ (1:s.nU)] = zU./xU.*dx[s.xU_bool] + r8./xU

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

function restoration_indices(s::Solver)
    p = s.n .+ (1:s.m)
    n = s.n + s.m .+ (1:s.m)
    zL = 1:s.nL
    zp = s.nL .+ (1:s.m)
    zn = s.nL + s.m .+ (1:s.m)
    zU = 1:s.nU
    xy = [s.idx.x...,(s.n + s.m + s.m .+ (1:s.m))...]

    RestorationIndices(p,n,zL,zp,zn,zU,xy)
end
