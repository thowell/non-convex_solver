function search_direction!(s::Solver)
    if s.opts.kkt_solve == :unreduced
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
    kkt_hessian_unreduced!(s)
    kkt_gradient_unreduced!(s)

    flag = inertia_correction!(s)

    s.δ[s.idx.x] .= s.δw
    s.δ[s.idx.λ] .= -s.δc
    s.d .= -(s.H + Diagonal(s.δ))\s.h

    if flag
        iterative_refinement(s.d,s)
    end

    s.δw = 0.
    s.δc = 0.
    return nothing
end
