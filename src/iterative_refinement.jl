function iterative_refinement(d,s::Solver; verbose=false)
    s.d_copy = copy(d)
    iter = 0
    s.res = -s.h - s.H*d

    res_norm = norm(s.res,Inf)

    while iter < s.opts.max_iterative_refinement && res_norm > s.opts.ϵ_iterative_refinement

        s.Δ .= (s.H+Diagonal(s.δ))\s.res
        d .+= s.Δ
        s.res = -s.h - s.H*d

        res_norm = norm(s.res,Inf)

        iter += 1
    end

    if res_norm < s.opts.ϵ_iterative_refinement
        verbose ? println("iterative refinement success: $(res_norm)") : nothing
        return true
    else
        d .= s.d_copy
        verbose ? println("iterative refinement failure: $(res_norm)") : nothing
        return false
    end
end
