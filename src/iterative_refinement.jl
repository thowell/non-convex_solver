function iterative_refinement(d,s::Solver; verbose=true)
    s.d_copy .= copy(d)
    iter = 0
    s.res .= -s.h - s.H*d

    res_norm = norm(s.res,Inf)
    res_norm_init = copy(res_norm)
    verbose ? println("init res: $(res_norm), δw: $(s.δw), δc: $(s.δc)") : nothing

    while (iter < s.opts.max_iterative_refinement && res_norm > s.opts.ϵ_iterative_refinement) || iter < s.opts.min_iterative_refinement
        if s.opts.kkt_solve == :unreduced
            s.Δ .= (s.H+Diagonal(s.δ))\s.res
        elseif s.opts.kkt_solve == :symmetric
            s.res[s.idx.xL] .+= s.res[s.idx.zL]./s.ΔxL
            s.res[s.idx.xU] .-= s.res[s.idx.zU]./s.ΔxU

            s.Δ[s.idx.xy] .= ma57_solve(s.LBL,s.res[s.idx.xy])
            s.Δ[s.idx.zL] .= -s.σL.*s.Δ[s.idx.xL] + s.res[s.idx.zL]./s.ΔxL
            s.Δ[s.idx.zU] .= s.σU.*s.Δ[s.idx.xU] + s.res[s.idx.zU]./s.ΔxU
        end

        d .+= s.Δ
        s.res .= -s.h - s.H*d

        res_norm = norm(s.res,Inf)

        iter += 1
    end

    if res_norm < s.opts.ϵ_iterative_refinement
        verbose ? println("iterative refinement success: $(res_norm), iter: $iter") : nothing #, cond: $(cond(Array(s.H+Diagonal(s.δ)))), rank: $(rank(Array(s.H+Diagonal(s.δ))))") : nothing
        return true
    else
        d .= s.d_copy
        verbose ? println("iterative refinement failure: $(res_norm), iter: $iter") : nothing #, cond: $(cond(Array(s.H+Diagonal(s.δ)))), rank: $(rank(Array(s.H+Diagonal(s.δ))))") : nothing
        return false
    end
end
