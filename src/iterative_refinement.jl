function iterative_refinement(d,s::Solver; verbose=true)
    s.d_copy = copy(d)
    iter = 0
    s.res = -s.h - s.H*d

    res_norm = norm(s.res,Inf)
    res_norm_init = copy(res_norm)
    println("init res: $(res_norm), δw: $(s.δw), δc: $(s.δc)")

    while (iter < s.opts.max_iterative_refinement && res_norm > s.opts.ϵ_iterative_refinement) || iter < s.opts.min_iterative_refinement
        if s.opts.kkt_solve == :unreduced
            s.Δ .= (s.H+Diagonal(s.δ))\s.res
        elseif s.opts.kkt_solve == :symmetric
            r̄ = copy(s.res)
            r̄3 = r̄[s.model.n+s.model.m .+ (1:s.nL)]
            r̄4 = r̄[s.model.n+s.model.m+s.nL .+ (1:s.nU)]
            r̄[(1:s.model.n)[s.xL_bool]] += r̄3./((s.x - s.xL)[s.xL_bool])
            r̄[(1:s.model.n)[s.xU_bool]] -= r̄4./((s.xU - s.x)[s.xU_bool])

            LBL = Ma57(s.H_sym + Diagonal(s.δ[1:(s.model.n+s.model.m)]))
            ma57_factorize(LBL)

            s.Δ[1:(s.model.n+s.model.m)] .= ma57_solve(LBL,r̄[1:(s.model.n+s.model.m)])
            s.Δ[(s.model.n+s.model.m) .+ (1:s.nL)] .= -s.zL./((s.x - s.xL)[s.xL_bool]).*s.Δ[1:s.model.n][s.xL_bool] + r̄3./((s.x - s.xL)[s.xL_bool])
            s.Δ[(s.model.n+s.model.m+s.nL) .+ (1:s.nU)] .= s.zU./((s.xU - s.x)[s.xU_bool]).*s.Δ[1:s.model.n][s.xU_bool] + r̄4./((s.xU - s.x)[s.xU_bool])
        end

        d .+= s.Δ
        s.res = -s.h - s.H*d

        res_norm = norm(s.res,Inf)

        iter += 1
    end

    if res_norm < s.opts.ϵ_iterative_refinement || res_norm < res_norm_init
        verbose ? println("iterative refinement success: $(res_norm), cond: $(cond(Array(s.H+Diagonal(s.δ)))), rank: $(rank(Array(s.H+Diagonal(s.δ))))") : nothing
        return true
    else
        d .= s.d_copy
        verbose ? println("iterative refinement failure: $(res_norm), cond: $(cond(Array(s.H+Diagonal(s.δ)))), rank: $(rank(Array(s.H+Diagonal(s.δ))))") : nothing
        return false
    end
end
