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
    elseif s.opts.kkt_solve == :slack
        search_direction_slack!(s)
    else
        error("KKT solve not implemented")
    end
    return small_search_direction(s)
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
    s.h[s.idx.zL] = s.zL.*s.ΔxL .- s.μ
    s.h[s.idx.zU] = s.zU.*s.ΔxU .- s.μ
    return nothing
end

function search_direction_fullspace!(s::Solver)
    kkt_hessian_symmetric!(s)
    inertia_correction!(s,restoration=s.restoration)

    kkt_hessian_fullspace!(s)

    s.d .= lu(s.H + Diagonal(s.δ))\(-s.h)

    s.opts.iterative_refinement && iterative_refinement(s.d,s)

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
    s.h_sym[s.idx.xL] .+= view(s.h,s.idx.zL)./(s.ΔxL .- s.δc)
    s.h_sym[s.idx.xU] .-= view(s.h,s.idx.zU)./(s.ΔxU .- s.δc)
    s.h_sym[s.idx.y] .= copy(view(s.h,s.idx.y))

    return nothing
end

function search_direction_symmetric!(s::Solver)
    kkt_hessian_symmetric!(s)
    kkt_gradient_symmetric!(s)

    inertia_correction!(s,restoration=s.restoration)
    s.dxy .= ma57_solve(s.LBL, -s.h_sym)
    # F = inertia_correction_qdldl(s)
    # s.dxy .= QDLDL.solve(F,-s.h_sym)

    s.dzL .= -s.σL.*s.dxL - s.zL + s.μ./(s.ΔxL .- s.δc)
    s.dzU .= s.σU.*s.dxU - s.zU + s.μ./(s.ΔxU .- s.δc)

    if s.opts.iterative_refinement
        kkt_hessian_fullspace!(s)
        iterative_refinement(s.d,s)
    end

    return nothing
end

function kkt_hessian_slack!(s::Solver)
    n = s.model_opt.n
    m = s.model_opt.m
    mI = s.model.mI
    mE = s.model.mE
    mA = s.model.mA
    nL = s.model_opt.nL
    nU = s.model_opt.nU
    idx = s.idx

    view(s.H_slack,1:n,1:n) .= s.∇²L[1:n,1:n]
    view(s.H_slack,CartesianIndex.(idx.xL[1:nL],idx.xL[1:nL])) .+= view(s.σL,1:nL)
    view(s.H_slack,CartesianIndex.(idx.xU[1:nU],idx.xU[1:nU])) .+= view(s.σU,1:nU)

    s.H_slack[1:n,n .+ (1:m)] .= get_∇c(s.model)[1:m,1:n]'
    s.H_slack[n .+ (1:m),1:n] .= get_∇c(s.model)[1:m,1:n]

    ΔxL = view(s.ΔxL,1:nL)
    ΔsL = view(s.ΔxL,nL .+ (1:mI))
    ΔxU = s.ΔxU
    zL = view(s.zL,1:nL)
    zS = view(s.zL,nL .+ (1:mI))
    zU = s.zU
    view(s.H_slack,CartesianIndex.(n .+ (1:mI),n .+ (1:mI))) .= -(ΔsL .- s.δc)./zS
    view(s.H_slack,CartesianIndex.(n+mI+mE .+ (1:mA),n+mI+mE .+ (1:mA))) .= -1.0/(s.ρ + s.δw)

    return nothing
end

function kkt_gradient_slack!(s::Solver)
    n = s.model_opt.n
    m = s.model_opt.m
    mI = s.model.mI
    mE = s.model.mE
    mA = s.model.mA
    nL = s.model_opt.nL
    nU = s.model_opt.nU
    idx = s.idx

    ΔxL = view(s.ΔxL,1:nL)
    ΔsL = view(s.ΔxL,nL .+ (1:mI))
    ΔxU = s.ΔxU
    zL = view(s.zL,1:nL)
    zS = view(s.zL,nL .+ (1:mI))
    zU = s.zU

    s.h_slack[1:n] .= copy(s.hx)
    s.h_slack[idx.xL[1:nL]] .+= s.hzL./(ΔxL .- s.δc)
    s.h_slack[idx.xU[1:nU]] .-= s.hzU./(ΔxU .- s.δc)
    s.h_slack[n .+ (1:mI)] .= s.hyI + ((ΔsL .- s.δc).*s.hs + s.hzs)./zS
    s.h_slack[n+mI .+ (1:mE)] .= copy(s.hyE)
    s.h_slack[n+mI+mE .+ (1:mA)] .= s.hyA + 1.0/(s.ρ + s.δw)*s.hr

    return nothing
end

function search_direction_slack!(s::Solver)
    kkt_hessian_slack!(s)
    kkt_gradient_slack!(s)

    n = s.model_opt.n
    m = s.model_opt.m
    mI = s.model.mI
    mE = s.model.mE
    mA = s.model.mA
    nL = s.model_opt.nL
    nU = s.model_opt.nU
    idx = s.idx

    ΔxL = view(s.ΔxL,1:nL)
    ΔsL = view(s.ΔxL,nL .+ (1:mI))
    ΔxU = s.ΔxU
    zL = view(s.zL,1:nL)
    zS = view(s.zL,nL .+ (1:mI))
    zU = s.zU

    inertia_correction_slack!(s)
    s._dxy .= ma57_solve(s.LBL_slack,-s.h_slack)

    s.dr .= 1.0/(s.ρ + s.δw)*(s.dyA - s.hr)
    s._dzL .= -(zL.*view(s.dxL,1:nL) + s.hzL)./(ΔxL .- s.δc)
    s.dzU .= (zU.*view(s.dx,1:nU) - s.hzU)./(ΔxU .- s.δc)


    s.dzs .= -s.dyI + s.hs
    s.ds .= -((ΔsL .- s.δc).*s.dzs + s.hzs)./zS

    # Is = Matrix(I,mI,mI)
    # tmp = [s.δw*Is -Is; Diagonal(zS) Diagonal(ΔsL .- s.δc)]\[-s.hs + s.dyI; -s.hzs]
    # s.ds .= tmp[1:mI]
    # s.dzs .= tmp[mI .+ (1:mI)]

    if s.opts.iterative_refinement
        kkt_hessian_fullspace!(s)
        iterative_refinement_slack(s.d,s)
    end
    return nothing
end
