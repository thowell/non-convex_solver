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
    s.mA != 0 && (s.cA .+= 1.0/s̄.ρ*(s̄.λ - s̄.yA))

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

function RestorationSolver(s::Solver;reformulate=false)
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
    s̄ = Solver(x̄,_model,cI_idx=s.cI_idx,cA_idx=s.cA_idx,reformulate=reformulate,opts=opts)
    s̄.DR = spzeros(s.n,s.n)
    s̄.idx_r = restoration_indices(s̄,s)
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
        ∇²f[s.idx.x,s.idx.x] = ζ*DR'*DR
        return nothing
    end

    s̄.f_func = f_func
    s̄.∇f_func! = ∇f_func!
    s̄.∇²f_func! = ∇²f_func!

    return nothing
end

function update_restoration_constraints!(s̄::Solver,s::Solver)
    function c_func!(c,x,model::AbstractModel)
        s.c_func!(c,view(x,s.idx.x),s.model)
        c .-= view(x,s̄.idx_r.p)
        c .+= view(x,s̄.idx_r.n)
        return nothing
    end

    function ∇c_func!(∇c,x,model::AbstractModel)
        s.∇c_func!(view(∇c,1:s.m,s.idx.x),view(x,s.idx.x),s.model)
        ∇c[CartesianIndex.(1:s.m,s̄.idx_r.p)] .= -1.0
        ∇c[CartesianIndex.(1:s.m,s̄.idx_r.n)] .= 1.0
        return nothing
    end

    function ∇²cy_func!(∇²cy,x,y,model::AbstractModel)
        s.∇²cy_func!(view(∇²cy,s.idx.x,s.idx.x),view(x,s.idx.x),y,s.model)
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
    kkt_gradient_fullspace!(s̄)
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
    s.Hv_sym.xy .= view(s̄.∇c,1:s.m,s.idx.x)'
    s.Hv_sym.yx .= view(s.∇c,1:s.m,s.idx.x)
    update!(s.Hv_sym.yy,-1.0./view(s̄.σL,s̄.idx_r.zLp) - 1.0./view(s̄.σL,s̄.idx_r.zLn))
    add_update!(s.Hv_sym.yAyA,-1.0/s̄.ρ)
    return nothing
end

function kkt_gradient_symmetric_restoration!(s̄::Solver,s::Solver)
    s.h_sym[s.idx.x] = s̄.h[s.idx.x]
    s.h_sym[s.idx.xL] += (view(s̄.h,s̄.idx.zL)./s̄.ΔxL)[1:s.nL]
    s.h_sym[s.idx.xU] -= (view(s̄.h,s̄.idx.zU)./s̄.ΔxU)[1:s.nU]
    s.h_sym[s.idx.y] = view(s̄.h,s̄.idx.y) + 1.0./view(s̄.σL,s̄.idx_r.zLp).*view(s̄.h,s̄.idx_r.p) -1.0./view(s̄.σL,s̄.idx_r.zLn).*view(s̄.h,s̄.idx_r.n) + view(s̄.h,s̄.idx_r.zp)./view(s̄.zL,s̄.idx_r.zLp) - view(s̄.h,s̄.idx_r.zn)./view(s̄.zL,s̄.idx_r.zLn)
    s.h_sym[s.idx.yA] += 1.0/s̄.ρ*(s̄.λ - s̄.yA)
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
    p = s.n .+ (1:s.m)
    n = s.n + s.m .+ (1:s.m)
    zL = s̄.n + s̄.m .+ (1:s.nL)
    zp = s̄.n + s̄.m + s.nL .+ (1:s.m)
    zn = s̄.n + s̄.m + s.nL + s.m .+ (1:s.m)
    zU = s̄.n + s̄.m + s.nL + s.m + s.m .+ (1:s.nU)
    xy = [s.idx.x...,(s̄.n .+ (1:s̄.m))...]

    zLL = 1:s.nL
    zLp = s.nL .+ (1:s.m)
    zLn = s.nL + s.m .+ (1:s.m)
    zUU = 1:s.nU
    RestorationIndices(p,n,zL,zp,zn,zU,xy,zLL,zLp,zLn,zUU)
end
