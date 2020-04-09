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
    s.H[s.idx.x,s.idx.x] .= s.W
    s.H[s.idx.x,s.idx.λ] .= s.A'
    s.H[s.idx.λ,s.idx.x] .= s.A
    s.H[CartesianIndex.(s.idx.xL,s.idx.zL)] .= -1.0
    s.H[CartesianIndex.(s.idx.xU,s.idx.zU)] .= 1.0
    s.H[CartesianIndex.(s.idx.zL,s.idx.xL)] .= s.zL
    s.H[CartesianIndex.(s.idx.zU,s.idx.xU)] .= -1.0*s.zU
    s.H[CartesianIndex.(s.idx.zL,s.idx.zL)] .= (s.x - s.xL)[s.xL_bool]
    s.H[CartesianIndex.(s.idx.zU,s.idx.zU)] .= (s.xU - s.x)[s.xU_bool]
    return nothing
end

function kkt_gradient_unreduced!(s::Solver)
    s.h[s.idx.x] = s.∇L
    s.h[s.idx.λ] = s.c
    s.h[s.idx.zL] = s.zL.*((s.x - s.xL)[s.xL_bool]) .- s.μ
    s.h[s.idx.zU] = s.zU.*((s.xU - s.x)[s.xU_bool]) .- s.μ
    return nothing
end

function search_direction_unreduced!(s::Solver)
    kkt_hessian_symmetric!(s)
    LBL = inertia_correction(s,restoration=s.restoration)

    kkt_hessian_unreduced!(s)
    kkt_gradient_unreduced!(s)
    s.d .= -(s.H + Diagonal(s.δ))\s.h

    iterative_refinement(s.d,s)

    return nothing
end

# symmetric KKT system
function kkt_hessian_symmetric!(s::Solver)
    s.ΣL[CartesianIndex.(s.idx.xL,s.idx.xL)] .= Diagonal((s.x - s.xL)[s.xL_bool])\s.zL
    s.ΣU[CartesianIndex.(s.idx.xU,s.idx.xU)] .= Diagonal((s.xU - s.x)[s.xU_bool])\s.zU

    s.H_sym[s.idx.x,s.idx.x] .= s.W + s.ΣL + s.ΣU
    s.H_sym[s.idx.x,s.idx.λ] .= s.A'
    s.H_sym[s.idx.λ,s.idx.x] .= s.A

    return nothing
end

function kkt_gradient_symmetric!(s::Solver)
    s.h_sym[s.idx.x] = s.∇φ + s.A'*s.λ
    s.h_sym[s.idx.λ] = s.c

    return nothing
end

function search_direction_symmetric!(s::Solver)
    kkt_hessian_symmetric!(s)
    kkt_gradient_symmetric!(s)

    LBL = inertia_correction(s,restoration=s.restoration)

    s.d[1:(s.model.n+s.model.m)] = ma57_solve(LBL, -s.h_sym)
    s.d[s.idx.zL] = -(Diagonal((s.x - s.xL)[s.xL_bool])\Diagonal(s.zL))*s.d[s.idx.xL] - s.zL + Diagonal((s.x - s.xL)[s.xL_bool])\(s.μ*ones(s.nL))
    s.d[s.idx.zU] = (Diagonal((s.xU - s.x)[s.xU_bool])\Diagonal(s.zU))*s.d[s.idx.xU] - s.zU + Diagonal((s.xU - s.x)[s.xU_bool])\(s.μ*ones(s.nU))

    kkt_hessian_unreduced!(s)
    kkt_gradient_unreduced!(s)
    iterative_refinement(s.d,s)

    return nothing
end
