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
    s.H[s.idx.x,s.idx.x] .= s.∇²L
    s.H[s.idx.x,s.idx.y] .= s.∇c'
    s.H[s.idx.y,s.idx.x] .= s.∇c
    s.H[CartesianIndex.(s.idx.xL,s.idx.zL)] .= -1.0
    s.H[CartesianIndex.(s.idx.xU,s.idx.zU)] .= 1.0
    s.H[CartesianIndex.(s.idx.zL,s.idx.xL)] .= s.zL
    s.H[CartesianIndex.(s.idx.zU,s.idx.xU)] .= -1.0*s.zU
    s.H[CartesianIndex.(s.idx.zL,s.idx.zL)] .= s.ΔxL
    s.H[CartesianIndex.(s.idx.zU,s.idx.zU)] .= s.ΔxU
    s.H[CartesianIndex.(s.idx.y_al,s.idx.y_al)] .= -1.0/s.ρ
    return nothing
end

function kkt_gradient_unreduced!(s::Solver)
    s.h[s.idx.x] = s.∇L
    s.h[s.idx.y] = s.c
    s.h[s.idx.y_al] .+= 1.0/s.ρ*(s.λ - s.y_al)
    s.h[s.idx.zL] = s.zL.*s.ΔxL .- s.μ
    s.h[s.idx.zU] = s.zU.*s.ΔxU .- s.μ
    return nothing
end

function search_direction_unreduced!(s::Solver)
    kkt_hessian_symmetric!(s)
    inertia_correction!(s,restoration=s.restoration)

    kkt_hessian_unreduced!(s)
    kkt_gradient_unreduced!(s)
    s.d .= lu(s.H + Diagonal(s.δ))\(-s.h)

    s.opts.iterative_refinement ? iterative_refinement(s.d,s) : nothing

    return nothing
end

# symmetric KKT system
function kkt_hessian_symmetric!(s::Solver)
    s.H_sym[s.idx.x,s.idx.x] .= s.∇²L
    s.H_sym[s.idx.xL,s.idx.xL] .+= Diagonal(s.σL)
    s.H_sym[s.idx.xU,s.idx.xU] .+= Diagonal(s.σU)
    s.H_sym[s.idx.x,s.idx.y] .= s.∇c'
    s.H_sym[s.idx.y,s.idx.x] .= s.∇c
    s.H_sym[CartesianIndex.(s.idx.y_al,s.idx.y_al)] .= -1.0/s.ρ

    return nothing
end

function kkt_gradient_symmetric!(s::Solver)
    s.h_sym[s.idx.x] .= s.∇φ + s.∇c'*s.y - s.∇c_al'*(s.λ + s.ρ*s.c_al)
    s.h_sym[s.idx.y] .= s.c
    s.h_sym[s.idx.y_al] .+= 1.0/s.ρ*(s.λ - s.y_al)

    return nothing
end

function search_direction_symmetric!(s::Solver)
    kkt_hessian_symmetric!(s)
    kkt_gradient_symmetric!(s)

    inertia_correction!(s,restoration=s.restoration)

    s.d[s.idx.xy] .= ma57_solve(s.LBL, -s.h_sym)
    s.dzL .= -s.σL.*s.d[s.idx.xL] - s.zL + s.μ./s.ΔxL
    s.dzU .= s.σU.*s.d[s.idx.xU] - s.zU + s.μ./s.ΔxU

    kkt_hessian_unreduced!(s)
    kkt_gradient_unreduced!(s)
    s.opts.iterative_refinement ? iterative_refinement(s.d,s) : nothing

    return nothing
end
