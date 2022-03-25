"""
    iterative_refinement(d, s::Solver)

Use iterative refinement on the fullspace KKT system to improve the current step `d`.
"""
function iterative_refinement(d::Vector{T},s::Solver) where T
    s.d_copy .= d
    iter = 0
    s.res .= -s.h - s.H*d

    res_norm = norm(s.res,Inf)
    res_norm_init = copy(res_norm)

    while (iter < s.opts.max_iterative_refinement && res_norm > s.opts.ϵ_iterative_refinement) || iter < s.opts.min_iterative_refinement
        if s.opts.kkt_solve == :fullspace
            s.Δ .= (s.H+Diagonal(s.δ))\s.res
        elseif s.opts.kkt_solve == :symmetric
            s.res_xL .+= s.res_zL./(s.ΔxL .- s.δc)
            s.res_xU .-= s.res_zU./(s.ΔxU .- s.δc)

            # s.Δ_xy .= ma57_solve(s.LBL,Array(s.res_xy))
            solve!(s.linear_solver,s.Δ_xy,Array(s.res_xy))
            s.Δ_zL .= -s.σL.*s.Δ_xL + s.res_zL./(s.ΔxL .- s.δc)
            s.Δ_zU .= s.σU.*s.Δ_xU + s.res_zU./(s.ΔxU .- s.δc)
        end

        d .+= s.Δ
        s.res .= -s.h - s.H*d

        res_norm = norm(s.res,Inf)

        iter += 1
    end

    # @logmsg InnerLoop "res: $(round(res_norm_init, sigdigits=1)) -> $(round(res_norm, sigdigits=1))"

    if res_norm < s.opts.ϵ_iterative_refinement
        return true
    else
        d .= s.d_copy
        return false
    end
end
