"""
    search_direction!(s::Solver)

Compute the search direction `s.d` by solving the KKT system. Includes both inertia
correction and iterative refinement.
"""
function search_direction!(s::Solver)
    if s.opts.kkt_solve == :symmetric
        search_direction_symmetric!(s)
    elseif s.opts.kkt_solve == :fullspace
        search_direction_fullspace!(s)
    else
        error("KKT solve not implemented")
    end
    return small_search_direction(s)
end

function kkt_hessian_fullspace!(s::Solver)
    update!(s.Hv.xx,s.∇²L)
    update!(s.Hv.xy,s.∇c')
    update!(s.Hv.yx,s.∇c)
    update!(s.Hv.xLzL,-1.0)
    update!(s.Hv.xUzU,1.0)
    update!(s.Hv.zLxL,s.zL)
    update!(s.Hv.zUxU,-1.0*s.zU)
    update!(s.Hv.zLzL,s.ΔxL)
    update!(s.Hv.zUzU,s.ΔxU)
    # update!(s.Hv.yAyA,-1.0/s.ρ)
    view(s.H,CartesianIndex.(s.model.n+s.mI .+ (1:s.mA),s.idx.yA)) .= -1.0/s.ρ
    return nothing
end

function kkt_gradient_fullspace!(s::Solver)
    s.h[s.idx.x] = s.∇L
    s.h[s.idx.y] = s.c
    s.h[s.idx.yA] += 1.0/s.ρ*(s.λ - s.yA)
    s.h[s.idx.zL] = s.zL.*s.ΔxL .- s.μ
    s.h[s.idx.zU] = s.zU.*s.ΔxU .- s.μ
    return nothing
end

function search_direction_fullspace!(s::Solver)
    kkt_hessian_symmetric!(s)
    inertia_correction!(s,restoration=s.restoration)

    kkt_hessian_fullspace!(s)
    kkt_gradient_fullspace!(s)
    s.d .= lu(s.H + Diagonal(s.δ))\(-s.h)

    s.opts.iterative_refinement && iterative_refinement(s.d,s)

    return nothing
end

# symmetric KKT system
function kkt_hessian_symmetric!(s::Solver)
    update!(s.Hv_sym.xx, s.∇²L)
    add_update!(s.Hv_sym.xLxL, s.σL)
    add_update!(s.Hv_sym.xUxU, s.σU)
    update!(s.Hv_sym.xy, s.∇c')
    update!(s.Hv_sym.yx, s.∇c)
    update!(s.Hv_sym.yAyA, -1.0/s.ρ)
    return nothing
end

function kkt_gradient_symmetric!(s::Solver)
    s.h_sym[s.idx.x] = s.∇φ + s.∇c'*s.y - s.∇cA'*(s.λ + s.ρ*s.cA)
    s.h_sym[s.idx.y] = s.c
    s.h_sym[s.idx.yA] += 1.0/s.ρ*(s.λ - s.yA)

    return nothing
end

function search_direction_symmetric!(s::Solver)
    kkt_hessian_symmetric!(s)
    kkt_gradient_symmetric!(s)

    inertia_correction!(s,restoration=s.restoration)

    s.dxy .= ma57_solve(s.LBL, -s.h_sym)
    s.dzL .= -s.σL.*s.dxL - s.zL + s.μ./s.ΔxL
    s.dzU .= s.σU.*s.dxU - s.zU + s.μ./s.ΔxU

    if s.opts.iterative_refinement
        kkt_hessian_fullspace!(s)
        kkt_gradient_fullspace!(s)
        iterative_refinement(s.d,s)
    end

    return nothing
end
