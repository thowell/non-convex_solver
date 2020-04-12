function iterative_refinement(d,s::Solver; verbose=true)
    s.d_copy .= copy(d)
    iter = 0
    s.res .= -s.h - (s.H + Diagonal(s.δ0))*d

    res_norm = norm(s.res,Inf)
    res_norm_init = copy(res_norm)
    verbose ? println("init res: $(res_norm), δw: $(s.δw), δc: $(s.δc)") : nothing

    # println("s.res: $(s.res)")
    # println("s.h: $(s.h)")
    # println("s.H: $(Array(s.H))")
    # println("s.δ: $(s.δ)")
    while (iter < s.opts.max_iterative_refinement && res_norm > s.opts.ϵ_iterative_refinement) || iter < s.opts.min_iterative_refinement
        if s.opts.kkt_solve == :unreduced
            s.Δ .= (s.H+Diagonal(s.δ+s.δ0))\s.res
        elseif s.opts.kkt_solve == :symmetric
            s.res[s.idx.xL] .+= s.res[s.idx.zL]./((s.x - s.xL)[s.xL_bool])
            s.res[s.idx.xU] .-= s.res[s.idx.zU]./((s.xU - s.x)[s.xU_bool])

            s.Δ[s.idx.xλ] .= ma57_solve(s.LBL,s.res[s.idx.xλ])
            s.Δ[s.idx.zL] .= -s.zL./((s.x - s.xL)[s.xL_bool]).*s.Δ[s.idx.xL] + s.res[s.idx.zL]./((s.x - s.xL)[s.xL_bool])
            s.Δ[s.idx.zU] .= s.zU./((s.xU - s.x)[s.xU_bool]).*s.Δ[s.idx.xU] + s.res[s.idx.zU]./((s.xU - s.x)[s.xU_bool])
        end

        d .+= s.Δ
        s.res .= -s.h - (s.H+ Diagonal(s.δ0))*d

        res_norm = norm(s.res,Inf)

        iter += 1
    end

    if res_norm < s.opts.ϵ_iterative_refinement ##|| res_norm < res_norm_init
        verbose ? println("iterative refinement success: $(res_norm), iter: $iter, cond: $(cond(Array(s.H+Diagonal(s.δ)))), rank: $(rank(Array(s.H+Diagonal(s.δ))))") : nothing
        return true
    else
        d .= s.d_copy
        verbose ? println("iterative refinement failure: $(res_norm), iter: $iter, cond: $(cond(Array(s.H+Diagonal(s.δ)))), rank: $(rank(Array(s.H+Diagonal(s.δ))))") : nothing
        return false
    end
end
