"""
    iterative_refinement(d, s::Solver)

Use iterative refinement on the fullspace KKT system to improve the current step `d`.
"""
function iterative_refinement(d::Vector{T}, s::Solver) where T
    s.d_copy .= d
    iter = 0
    s.res .= -s.h - s.H*d

    res_norm = norm(s.res,Inf)

    while (iter < s.options.max_iterative_refinement && res_norm > s.options.iterative_refinement_tolerance) || iter < s.options.min_iterative_refinement
        if s.options.linear_solve_type == :fullspace
            s.Δ .= (s.H+Diagonal(s.regularization))\s.res
        elseif s.options.linear_solve_type == :symmetric
            s.res_xL .+= s.res_zL./(s.ΔxL .- s.dual_regularization)
            s.res_xU .-= s.res_zU./(s.ΔxU .- s.dual_regularization)

            solve!(s.linear_solver,s.Δ_xy,Array(s.res_xy))
            s.Δ_zL .= -s.σL.*s.Δ_xL + s.res_zL./(s.ΔxL .- s.dual_regularization)
            s.Δ_zU .= s.σU.*s.Δ_xU + s.res_zU./(s.ΔxU .- s.dual_regularization)
        end

        d .+= s.Δ
        s.res .= -s.h - s.H*d

        res_norm = norm(s.res,Inf)

        iter += 1
    end

    if res_norm < s.options.iterative_refinement_tolerance
        return true
    else
        d .= s.d_copy
        return false
    end
end
