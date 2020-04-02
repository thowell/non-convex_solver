function iterative_refinement(d_,s::Solver; verbose=false)
    kkt_hessian_unreduced!(s)
    kkt_gradient_unreduced!(s)

    # kkt_hessian_symmetric!(s)
    # kkt_gradient_symmetric!(s)

    dd = copy(d_)
    iter = 0
    res = -s.hu - s.Hu*dd

    res_norm = norm(res,1)

    while iter < s.opts.max_iterative_refinement && norm(res,Inf) > s.opts.ϵ_iterative_refinement
        d = zero(d_)
        # r̄ = copy(res)
        # r̄3 = r̄[s.n+s.m .+ (1:s.nL)]
        # r̄4 = r̄[s.n+s.m+s.nL .+ (1:s.nU)]
        # r̄[(1:s.n)[s.xL_bool]] += r̄3./((s.x - s.xL)[s.xL_bool])
        # r̄[(1:s.n)[s.xU_bool]] -= r̄4./((s.xU - s.x)[s.xU_bool])
        #
        # LBL = Ma57(s.H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))
        # ma57_factorize(LBL)
        # d[1:(s.n+s.m)] = ma57_solve(LBL,r̄[1:(s.n+s.m)])
        #
        # d[(s.n+s.m) .+ (1:s.nL)] = -s.zL./((s.x - s.xL)[s.xL_bool]).*d[1:s.n][s.xL_bool] + r̄3./((s.x - s.xL)[s.xL_bool])
        # d[(s.n+s.m+s.nL) .+ (1:s.nU)] = s.zU./((s.xU - s.x)[s.xU_bool]).*d[1:s.n][s.xU_bool] + r̄4./((s.xU - s.x)[s.xU_bool])

        d = (s.Hu+Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m);zeros(s.nL+s.nU)]))\res
        dd .+= d
        res = -s.hu - s.Hu*dd
        # println("d: $d")

        iter += 1
    end

    if norm(res,Inf) < s.opts.ϵ_iterative_refinement #|| norm(res,1) < res_norm
        s.d .= dd
        println("iterative refinement success: $(norm(res,Inf))")
        return true
    else
        println("iterative refinement failure: $(norm(res,Inf))")
        # println("δ: $(δ)")
        false
    end
end

function iterative_refinement_soc(d_,s::Solver; verbose=false)
    kkt_hessian_unreduced!(s)
    kkt_gradient_unreduced!(s)

    s.hu[s.n .+ (1:s.m)] = s.c_soc

    # kkt_hessian_symmetric!(s)
    # kkt_gradient_symmetric!(s)

    dd = copy(d_)
    iter = 0
    res = -s.hu - s.Hu*dd

    res_norm = norm(res,1)

    while iter < s.opts.max_iterative_refinement && norm(res,Inf) > s.opts.ϵ_iterative_refinement
        d = zero(d_)
        # r̄ = copy(res)
        # r̄3 = r̄[s.n+s.m .+ (1:s.nL)]
        # r̄4 = r̄[s.n+s.m+s.nL .+ (1:s.nU)]
        # r̄[(1:s.n)[s.xL_bool]] += r̄3./((s.x - s.xL)[s.xL_bool])
        # r̄[(1:s.n)[s.xU_bool]] -= r̄4./((s.xU - s.x)[s.xU_bool])
        #
        # LBL = Ma57(s.H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))
        # ma57_factorize(LBL)
        # d[1:(s.n+s.m)] = ma57_solve(LBL,r̄[1:(s.n+s.m)])
        #
        # d[(s.n+s.m) .+ (1:s.nL)] = -s.zL./((s.x - s.xL)[s.xL_bool]).*d[1:s.n][s.xL_bool] + r̄3./((s.x - s.xL)[s.xL_bool])
        # d[(s.n+s.m+s.nL) .+ (1:s.nU)] = s.zU./((s.xU - s.x)[s.xU_bool]).*d[1:s.n][s.xU_bool] + r̄4./((s.xU - s.x)[s.xU_bool])

        d = (s.Hu+Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m);zeros(s.nL+s.nU)]))\res
        dd .+= d
        res = -s.hu - s.Hu*dd
        # println("d: $d")

        iter += 1
    end

    if norm(res,Inf) < s.opts.ϵ_iterative_refinement #|| norm(res,1) < res_norm
        s.d .= dd
        println("iterative refinement success: $(norm(res,Inf))")
        return true
    else
        println("iterative refinement failure: $(norm(res,Inf))")
        # println("δ: $(δ)")
        false
    end
end

function iterative_refinement(x_,A,δ,b,n,m; max_iter=10,ϵ=1.0e-16,verbose=false)

    x = copy(x_)
    iter = 0
    res = b - A*x

    while iter < max_iter && norm(res,Inf) > ϵ
        x .+= (A+Diagonal(δ))\res
        # println("x: $x")

        res = b - A*x
        iter += 1
    end

    if norm(res,Inf) < ϵ
        x_ .= x
        println("iterative refinement success")
        return true
    else
        println("iterative refinement failure: $(norm(res,Inf))")
        println("δ: $(δ)")
        false
    end
end

function iterative_refinement(d_,s̄::Solver,s::Solver; verbose=false)
    kkt_hessian_unreduced!(s̄)
    kkt_gradient_unreduced!(s̄)
    # search_direction_restoration!(s̄,s)

    idx = [(1:s.n)...,(s.n+2s.m .+ (1:s.m))...]

    x = s̄.x[1:s.n]
    p = s̄.x[s.n .+ (1:s.m)]
    n = s̄.x[s.n + s.m .+ (1:s.m)]
    λ = s̄.λ
    zL = s̄.zL[1:s.nL]
    zp = s̄.zL[s.nL .+ (1:s.m)]
    zn = s̄.zL[s.nL + s.m .+ (1:s.m)]
    zU = s̄.zU[1:s.nU]

    xL = (x - s.xL)[s.xL_bool]
    xU = (s.xU - x)[s.xU_bool]

    dd = copy(d_)
    iter = 0
    res = -s̄.hu - s̄.Hu*dd

    res_norm = norm(res,1)

    while iter < s.opts.max_iterative_refinement && norm(res,Inf) > s.opts.ϵ_iterative_refinement
        d = zero(d_)
        # r = copy(res)
        # r1 = r[1:s.n]
        # r2 = r[s.n .+ (1:s.m)]
        # r3 = r[s.n + s.m .+ (1:s.m)]
        # r4 = r[s.n + s.m + s.m .+ (1:s.m)]
        # r5 = r[s.n + s.m + s.m + s.m .+ (1:s.nL)]
        # r6 = r[s.n + s.m + s.m + s.m + s.nL .+ (1:s.m)]
        # r7 = r[s.n + s.m + s.m + s.m + s.nL + s.m .+ (1:s.m)]
        # r8 = r[s.n + s.m + s.m + s.m + s.nL + s.m + s.m .+ (1:s.nU)]
        #
        # r̄1 = copy(r1)
        # r̄1[s.xL_bool] += r5./xL
        # r̄1[s.xU_bool] -= r8./xU
        # r̄4 = copy(r4)
        # r̄4 .+= p./zp.*r2 + r6./zp - n./zn.*r3 - r7./zn
        #
        #
        #
        # LBL = Ma57(s.H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))
        # # LBL = Ma57([s.W s̄.A[:,1:s.n]'; s̄.A[:,1:s.n] (-Diagonal(p./zp) - Diagonal(n./zn))])
        # ma57_factorize(LBL)
        # d[idx] = ma57_solve(LBL,[r̄1;r̄4])
        #
        # dx = d[1:s.n]
        # dλ = d[s.n+2s.m .+ (1:s.m)]
        #
        # d[s.n .+ (1:s.m)] = -p.*(-dλ - r2)./zp + r6./zp
        # d[s.n + s.m .+ (1:s.m)] = -n.*(dλ - r3)./zn + r7./zn
        # d[s.n + 3s.m .+ (1:s.nL)] = -zL./xL.*dx[s.xL_bool] + r5./xL
        # d[s.n + 3s.m + s.nL .+ (1:s.m)] = -dλ - r2
        # d[s.n + 4s.m + s.nL .+ (1:s.m)] = dλ - r3
        # d[s.n + 5s.m + s.nL .+ (1:s.nU)] = zU./xU.*dx[s.xU_bool] + r8./xU

        d = (s̄.Hu + Diagonal([s.δw*ones(s.n);zeros(2s.m);-s.δc*ones(s.m);zeros(s̄.nL+s̄.nU)]))\res

        dd .+= d
        res = -s̄.hu - s̄.Hu*dd
        # println("d: $d")

        iter += 1
    end

    if norm(res,Inf) < s.opts.ϵ_iterative_refinement #|| norm(res,1) < res_norm
        d_ .= dd
        println("iterative refinement success")
        return true
    else
        println("iterative refinement failure: $(norm(res,Inf))")
        # println("δ: $(δ)")
        false
    end
end
