function search_direction!(s::Solver)
    if s.opts.kkt_solve == :unreduced
        search_direction_unreduced!(s)
    else
        error("KKT solve not implemented")
    end
    return small_search_direction(s)
end

function kkt_hessian_unreduced!(s::Solver)
    s.Hu[1:s.n,1:s.n] .= s.W
    s.Hu[1:s.n,s.n .+ (1:s.m)] .= s.A'
    s.Hu[s.n .+ (1:s.m),1:s.n] .= s.A
    s.Hu[(1:s.n)[s.xL_bool],(s.n+s.m) .+ (1:s.nL)] .= -1.0*Matrix(I,s.nL,s.nL)
    s.Hu[(1:s.n)[s.xU_bool],(s.n+s.m+s.nL) .+ (1:s.nU)] .= 1.0*Matrix(I,s.nU,s.nU)
    s.Hu[(s.n+s.m) .+ (1:s.nL),(1:s.n)[s.xL_bool]] .= Diagonal(s.zL)
    s.Hu[(s.n+s.m+s.nL) .+ (1:s.nU),(1:s.n)[s.xU_bool]] .= -1.0*Diagonal(s.zU)
    s.Hu[(s.n+s.m) .+ (1:s.nL),(s.n+s.m) .+ (1:s.nL)] .= Diagonal((s.x - s.xL)[s.xL_bool])
    s.Hu[(s.n+s.m+s.nL) .+ (1:s.nU),(s.n+s.m+s.nL) .+ (1:s.nU)] .= Diagonal((s.xU - s.x)[s.xU_bool])
    return nothing
end

function kkt_gradient_unreduced!(s::Solver)
    s.hu[1:s.n] = s.∇L
    s.hu[s.n .+ (1:s.m)] = s.c
    s.hu[(s.n+s.m) .+ (1:s.nL)] = s.zL.*((s.x - s.xL)[s.xL_bool]) .- s.μ
    s.hu[(s.n+s.m+s.nL) .+ (1:s.nU)] = s.zU.*((s.xU - s.x)[s.xU_bool]) .- s.μ
    return nothing
end

function search_direction_unreduced!(s::Solver)
    kkt_hessian_unreduced!(s)
    kkt_gradient_unreduced!(s)

    flag = inertia_correction!(s)

    s.δ[1:s.n] .= s.δw
    s.δ[s.n .+ (1:s.m)] .= -s.δc
    s.d .= -(s.Hu + Diagonal(s.δ))\s.hu

    if flag
        iterative_refinement(s.d,s)
    end

    s.δw = 0.
    s.δc = 0.
    return nothing
end
