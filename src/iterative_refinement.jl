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
            r̄3 = r̄[s.idx.zL]
            r̄4 = r̄[s.idx.zU]
            r̄[s.idx.xL] .+= r̄3./((s.x - s.xL)[s.xL_bool])
            r̄[s.idx.xU] .-= r̄4./((s.xU - s.x)[s.xU_bool])

            s.Δ[s.idx.xλ] .= ma57_solve(s.LBL,r̄[s.idx.xλ])
            s.Δ[s.idx.zL] .= -s.zL./((s.x - s.xL)[s.xL_bool]).*s.Δ[s.idx.xL] + r̄3./((s.x - s.xL)[s.xL_bool])
            s.Δ[s.idx.zU] .= s.zU./((s.xU - s.x)[s.xU_bool]).*s.Δ[s.idx.xU] + r̄4./((s.xU - s.x)[s.xU_bool])
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
