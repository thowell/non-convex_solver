function search_direction!(s::Solver)
    search_direction_sym!(s::Solver)
    return nothing
end

# symmetric KKT system

function kkt_hessian_sym!(s::Solver)
    ∇L(x) = s.∇f_func(x) + s.∇c_func(x)'*s.λ
    s.W .= ForwardDiff.jacobian(∇L,s.x)
    s.Σl[CartesianIndex.((1:s.n)[s.xl_bool],(1:s.n)[s.xl_bool])] .= s.zl./((s.x - s.xl)[s.xl_bool])
    s.Σu[CartesianIndex.((1:s.n)[s.xu_bool],(1:s.n)[s.xu_bool])] .= s.zu./((s.xu - s.x)[s.xu_bool])

    s.H[1:s.n,1:s.n] .= (s.W + s.Σl + s.Σu)
    s.H[1:s.n,s.n .+ (1:s.m)] .= s.A'
    s.H[s.n .+ (1:s.m),1:s.n] .= s.A

    return nothing
end

function kkt_gradient_sym!(s::Solver)
    s.h[1:s.n] .= s.∇φ + s.A'*s.λ
    s.h[s.n .+ (1:s.m)] .= s.c

    return nothing
end

function search_direction_sym!(s::Solver)
    kkt_hessian_sym!(s)
    kkt_gradient_sym!(s)

    s.d[1:(s.n+s.m)] .= -s.H\s.h
    s.d[(s.n+s.m) .+ (1:s.nl)] = -s.zl./((s.x - s.xl)[s.xl_bool]).*s.d[(1:s.n)[s.xl_bool]] - s.zl + s.μ./((s.x - s.xl)[s.xl_bool])
    s.d[(s.n+s.m+s.nl) .+ (1:s.nu)] .= s.zu./((s.xu - s.x)[s.xu_bool]).*s.d[(1:s.n)[s.xu_bool]] - s.zu + s.μ./((s.xu - s.x)[s.xu_bool])
    return nothing
end

# full KKT system

function kkt_hessian_full!(s::Solver)
    # ∇L(x) = s.∇f_func(x) + s.∇c_func(x)'*s.λ
    # s.W .= ForwardDiff.jacobian(∇L,s.x)
    # s.Σl[CartesianIndex.((1:s.n)[s.xl_bool],(1:s.n)[s.xl_bool])] .= s.zl./((s.x - s.xl)[s.xl_bool])
    # s.Σu[CartesianIndex.((1:s.n)[s.xu_bool],(1:s.n)[s.xu_bool])] .= s.zu./((s.xu - s.x)[s.xu_bool])
    #
    # s.H[1:s.n,1:s.n] .= (s.W + s.Σl + s.Σu)
    # s.H[1:s.n,s.n .+ (1:s.m)] .= s.A'
    # s.H[s.n .+ (1:s.m),1:s.n] .= s.A

    return nothing
end

function kkt_gradient_full!(s::Solver)
    # s.h[1:s.n] .= s.∇φ + s.A'*s.λ
    # s.h[s.n .+ (1:s.m)] .= s.c

    return nothing
end

function search_direction_full!(s::Solver)
    kkt_hessian_full!(s)
    kkt_gradient_full!(s)

    s.d .= -s.H\s.h
    return nothing
end
