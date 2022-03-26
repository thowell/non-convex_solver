"""
    search_direction!(s::Solver)

Compute the search direction `s.d` by solving the KKT system. Includes both inertia
correction and iterative refinement.
"""
function search_direction!(s::Solver)
    search_direction_symmetric!(s)
    return
end

function kkt_hessian_fullspace!(s::Solver)
    update!(s.Hv.xx,s.∇²L)
    update!(s.Hv.xy,get_∇c(s.model)')
    update!(s.Hv.yx,get_∇c(s.model))
    update!(s.Hv.xLzL,-1.0)
    update!(s.Hv.xUzU,1.0)
    update!(s.Hv.zLxL,s.zL)
    update!(s.Hv.zUxU,-1.0*s.zU)
    update!(s.Hv.zLzL,s.ΔxL)
    update!(s.Hv.zUzU,s.ΔxU)
    return nothing
end

function kkt_gradient_fullspace!(s::Solver)
    s.h[s.idx.x] = s.∇L
    s.h[s.idx.y] = s.c
    s.h[s.idx.zL] = s.zL.*s.ΔxL .- s.central_path
    s.h[s.idx.zU] = s.zU.*s.ΔxU .- s.central_path
    return nothing
end

function search_direction_fullspace!(s::Solver)
    kkt_hessian_symmetric!(s)
    inertia_correction!(s)

    kkt_hessian_fullspace!(s)

    s.d .= lu(s.H + Diagonal(s.regularization))\(-s.h)

    s.options.iterative_refinement && iterative_refinement(s.d,s)

    return nothing
end

# symmetric KKT system
function kkt_hessian_symmetric!(s::Solver)
    update!(s.Hv_sym.xx, s.∇²L)
    add_update!(s.Hv_sym.xLxL, s.σL)
    add_update!(s.Hv_sym.xUxU, s.σU)
    update!(s.Hv_sym.xy, get_∇c(s.model)')
    update!(s.Hv_sym.yx, get_∇c(s.model))
    return nothing
end

function kkt_gradient_symmetric!(s::Solver)
    s.h_sym[s.idx.x] .= copy(view(s.h,s.idx.x))
    s.h_sym[s.idx.xL] .+= view(s.h,s.idx.zL)./(s.ΔxL .- s.dual_regularization)
    s.h_sym[s.idx.xU] .-= view(s.h,s.idx.zU)./(s.ΔxU .- s.dual_regularization)
    s.h_sym[s.idx.y] .= copy(view(s.h,s.idx.y))

    return nothing
end

function search_direction_symmetric!(s::Solver)
    kkt_hessian_symmetric!(s)
    kkt_gradient_symmetric!(s)

    inertia_correction!(s)
    solve!(s.linear_solver,s.dxy,-s.h_sym)
    s.dzL .= -s.σL.*s.dxL - s.zL + s.central_path./(s.ΔxL .- s.dual_regularization)
    s.dzU .= s.σU.*s.dxU - s.zU + s.central_path./(s.ΔxU .- s.dual_regularization)

    if s.options.iterative_refinement
        kkt_hessian_fullspace!(s)
        iterative_refinement(s.d,s)
    end

    return nothing
end
