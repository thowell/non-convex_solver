"""
    search_direction!(s::Solver)

Compute the search direction `s.d` by solving the KKT system. Includes both inertia
correction and iterative refinement.
"""
function search_direction!(s::Solver)
    if s.opts.kkt_solve == :symmetric
        search_direction_symmetric!(s)
    elseif s.opts.kkt_solve == :unreduced
        search_direction_unreduced!(s)
    else
        error("KKT solve not implemented")
    end
    return small_search_direction(s)
end

function kkt_hessian_unreduced!(s::Solver)
    update!(s.Hv.xx,s.W)
    update!(s.Hv.xy,s.∇c')
    update!(s.Hv.xLzL,-1.0)
    update!(s.Hv.xUzU,1.0)

    update!(s.Hv.yx,s.∇c)
    s.mI != 0 && update!(view(s.H,CartesianIndex.(s.idx.yI,s.idx.s)),-1.0)
    s.mA != 0 && update!(view(s.H,CartesianIndex.(s.idx.yA,s.idx.r)),-1.0)

    update!(s.Hv.zLxL,s.zL)
    update!(s.Hv.zLzL,s.ΔxL)

    update!(s.Hv.zUxU,-1.0*s.zU)
    update!(s.Hv.zUzU,s.ΔxU)

    if s.mI != 0
        update!(view(s.H,CartesianIndex.(s.idx.s,s.idx.yI)),-1.0)
        update!(view(s.H,CartesianIndex.(s.idx.s,s.idx.zS)),-1.0)

        update!(view(s.H,CartesianIndex.(s.idx.zS,s.idx.s)),s.zS)
        update!(view(s.H,CartesianIndex.(s.idx.zS,s.idx.zS)),s.ΔsL)
    end

    if s.mA != 0
        update!(view(s.H,CartesianIndex.(s.idx.r,s.idx.yA)),-1.0/s.ρ)
        update!(view(s.H,CartesianIndex.(s.idx.r,s.idx.r)),1.0)
    end

    return nothing
end

function kkt_gradient_unreduced!(s::Solver)
    s.h[s.idx.primals] = s.∇L
    s.h[s.idx.y] = s.c
    s.mI != 0 && (s.h[s.idx.yI] -= s.s)
    s.mA != 0 && (s.h[s.idx.yA] -= s.r)
    s.h[s.idx.zL] = s.zL.*s.ΔxL .- s.μ
    s.h[s.idx.zU] = s.zU.*s.ΔxU .- s.μ
    s.mI != 0 && (s.h[s.idx.zS] = s.zS.*s.ΔsL .- s.μ)
    return nothing
end

function search_direction_unreduced!(s::Solver)
    kkt_hessian_symmetric!(s)
    inertia_correction!(s,restoration=s.restoration)

    kkt_hessian_unreduced!(s)
    kkt_gradient_unreduced!(s)
    s.d .= lu(s.H + Diagonal(s.δ))\(-s.h)

    s.opts.iterative_refinement && iterative_refinement(s.d,s)

    return nothing
end

# symmetric KKT system
function kkt_hessian_symmetric!(s::Solver)
    update!(s.Hv_sym.xx, s.W)
    add_update!(s.Hv_sym.xLxL, s.σL)
    add_update!(s.Hv_sym.xUxU, s.σU)
    update!(s.Hv_sym.xy, s.∇c')
    update!(s.Hv_sym.yx, s.∇c)
    s.mI != 0 && update!(view(s.H_sym,CartesianIndex.(s.idx.yI,s.idx.yI)),-s.ΔsL./s.zS)
    s.mA != 0 && update!(view(s.H_sym,CartesianIndex.(s.idx.yA,s.idx.yA)),-1.0/s.ρ)
    return nothing
end

function kkt_gradient_symmetric!(s::Solver)
    # s.h_sym[s.idx.x] = s.∇φ + s.∇c'*s.y - s.∇cA'*(s.λ + s.ρ*s.cA)
    # s.h_sym[s.idx.y] = s.c
    # s.h_sym[s.idx.yA] += 1.0/s.ρ*(s.λ - s.yA)

    kkt_gradient_unreduced!(s)

    s.h_sym[s.idx.x] = s.h[s.idx.x]
    s.h_sym[s.idx.xL] .+= s.h[s.idx.zL]./s.ΔxL
    s.h_sym[s.idx.xU] .-= s.h[s.idx.zU]./s.ΔxU

    s.h_sym[s.idx.yI] = s.h[s.idx.yI] + (s.ΔsL.*s.h[s.idx.s] + s.h[s.idx.zS])./s.zS
    s.h_sym[s.idx.yE] = s.h[s.idx.yE]
    s.h_sym[s.idx.yA] = s.h[s.idx.yA] + s.h[s.idx.r]
    return nothing
end

function search_direction_symmetric!(s::Solver)
    kkt_hessian_symmetric!(s)
    kkt_gradient_symmetric!(s)

    inertia_correction!(s,restoration=s.restoration)

    s.dxy .= ma57_solve(s.LBL, -s.h_sym)
    # s.dzL .= -s.σL.*s.dxL - s.zL + s.μ./s.ΔxL
    # s.dzU .= s.σU.*s.dxU - s.zU + s.μ./s.ΔxU
    s.dzL .= -s.σL.*s.dxL - s.h[s.idx.zL]./s.ΔxL
    s.dzU .= s.σU.*s.dxU - s.h[s.idx.zU]./s.ΔxU

    if s.mI != 0
        s.dzS .= -s.dy[s.cI_idx] + s.h[s.idx.s]
        s.ds .= -s.ΔsL./s.zS.*s.dzS - s.h[s.idx.zS]./s.zS
    end

    s.mA != 0 && (s.dr .= 1.0/s.ρ*s.dy[s.cA_idx] - s.h[s.idx.r])

    if s.opts.iterative_refinement
        kkt_hessian_unreduced!(s)
        kkt_gradient_unreduced!(s)
        iterative_refinement(s.d,s)
    end

    return nothing
end
