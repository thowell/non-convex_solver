"""
    iterative_refinement(d, s::Solver)

Use iterative refinement on the unreduced KKT system to improve the current step `d`.
"""
function iterative_refinement(d::Vector{T},s::Solver) where T
    s.d_copy .= d
    iter = 0
    s.res .= -s.h - s.H*d

    res_norm = norm(s.res,Inf)

    while (iter < s.opts.max_iterative_refinement && res_norm > s.opts.ϵ_iterative_refinement) || iter < s.opts.min_iterative_refinement
        if s.opts.kkt_solve == :unreduced
            s.Δ .= (s.H+Diagonal(s.δ))\s.res
        elseif s.opts.kkt_solve == :symmetric
            s.res_xL .+= s.res_zL./s.ΔxL
            s.res_xU .-= s.res_zU./s.ΔxU

            s.Δ_xy .= ma57_solve(s.LBL,s.res[s.idx.xy])
            s.Δ_zL .= -s.σL.*s.Δ_xL + s.res_zL./s.ΔxL
            s.Δ_zU .= s.σU.*s.Δ_xU + s.res_zU./s.ΔxU
        end

        d .+= s.Δ
        s.res .= -s.h - s.H*d

        res_norm = norm(s.res,Inf)

        iter += 1
    end

    if res_norm < s.opts.ϵ_iterative_refinement
        return true
    else
        d .= s.d_copy
        return false
    end
end
