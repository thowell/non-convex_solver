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

# symmetric KKT system
function kkt_hessian_symmetric!(s::Solver)
    s.ΣL[CartesianIndex.((1:s.n)[s.xL_bool],(1:s.n)[s.xL_bool])] .= s.zL./((s.x - s.xL)[s.xL_bool])
    s.ΣU[CartesianIndex.((1:s.n)[s.xU_bool],(1:s.n)[s.xU_bool])] .= s.zU./((s.xU - s.x)[s.xU_bool])

    s.H[1:s.n,1:s.n] .= s.W + s.ΣL + s.ΣU
    s.H[1:s.n,s.n .+ (1:s.m)] .= s.A'
    s.H[s.n .+ (1:s.m),1:s.n] .= s.A

    return nothing
end

function kkt_gradient_symmetric!(s::Solver)
    s.h[1:s.n] = s.∇φ + s.A'*s.λ
    s.h[s.n .+ (1:s.m)] = s.c

    return nothing
end

function search_direction_symmetric!(s::Solver)
    kkt_hessian_symmetric!(s)
    kkt_gradient_symmetric!(s)

    flag = inertia_correction_hsl!(s.H,s)

    LBL = Ma57(s.H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))
    ma57_factorize(LBL)
    s.d[1:(s.n+s.m)] = ma57_solve(LBL, -s.h)

    # s.d[1:(s.n+s.m)] = -Symmetric(s.H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))\s.h
    s.d[(s.n+s.m) .+ (1:s.nL)] = -s.zL./((s.x - s.xL)[s.xL_bool]).*s.d[(1:s.n)[s.xL_bool]] - s.zL + s.μ./((s.x - s.xL)[s.xL_bool])
    s.d[(s.n+s.m+s.nL) .+ (1:s.nU)] = s.zU./((s.xU - s.x)[s.xU_bool]).*s.d[(1:s.n)[s.xU_bool]] - s.zU + s.μ./((s.xU - s.x)[s.xU_bool])

    if flag
        iterative_refinement(s.d,s)
    end

    s.δw = 0.
    s.δc = 0.
    return nothing
end

# full KKT system

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
#
function search_direction_unreduced!(s::Solver)

    kkt_hessian_symmetric!(s)
    kkt_gradient_symmetric!(s)

    kkt_hessian_unreduced!(s)
    kkt_gradient_unreduced!(s)

    flag = inertia_correction_hsl!(s.H,s)



    # flag = inertia_correction_unreduced!(s.Hu,s,false)

    s.d .= lu(s.Hu + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m);zeros(s.nL+s.nU)]))\(-1.0*s.hu)

    if flag
        iterative_refinement(s.d,s.Hu,
            [s.δw*ones(s.n);-s.δc*ones(s.m);zeros(s.nL+s.nU)],-s.hu,s.n,s.m,
            max_iter=s.opts.max_iterative_refinement,ϵ=s.opts.ϵ_iterative_refinement)
    end
    s.δw = 0.
    s.δc = 0.
    return nothing
end
