function iterative_refinement(d,s::Solver; verbose=false)
    s.d_copy = copy(d)
    iter = 0
    s.res = -s.hu - s.Hu*d

    while iter < s.opts.max_iterative_refinement && norm(s.res,Inf) > s.opts.ϵ_iterative_refinement

        s.Δ .= (s.Hu+Diagonal(s.δ))\s.res
        d .+= s.Δ
        s.res = -s.hu - s.Hu*d

        iter += 1
    end

    if norm(s.res,Inf) < s.opts.ϵ_iterative_refinement #|| norm(res,1) < res_norm
        verbose ? println("iterative refinement success: $(norm(s.res,Inf))") : nothing
        return true
    else
        d .= s.d_copy
        verbose ? println("iterative refinement failure: $(norm(s.res,Inf))") : nothing
        return false
    end
end
