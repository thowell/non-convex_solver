function iterative_refinement(d_,s::Solver; verbose=false)
    kkt_hessian_unreduced!(s)
    kkt_gradient_unreduced!(s)

    # kkt_hessian_symmetric!(s)
    # kkt_gradient_symmetric!(s)

    d = copy(d_)
    iter = 0
    res = -s.hu - s.Hu*d

    res_norm = norm(res,1)

    while iter < s.opts.max_iterative_refinement && norm(res,Inf) > s.opts.ϵ_iterative_refinement
        r̄ = copy(res)
        r̄3 = r̄[s.n+s.m .+ (1:s.nL)]
        r̄4 = r̄[s.n+s.m+s.nL .+ (1:s.nU)]
        r̄[(1:s.n)[s.xL_bool]] += r̄3./((s.x - s.xL)[s.xL_bool])
        r̄[(1:s.n)[s.xU_bool]] -= r̄4./((s.xU - s.x)[s.xU_bool])

        LBL = Ma57(s.H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))
        ma57_factorize(LBL)
        d[1:(s.n+s.m)] += ma57_solve(LBL,r̄[1:(s.n+s.m)])

        d[(s.n+s.m) .+ (1:s.nL)] += -s.zL./((s.x - s.xL)[s.xL_bool]).*d[1:s.n][s.xL_bool] + r̄3./((s.x - s.xL)[s.xL_bool])
        d[(s.n+s.m+s.nL) .+ (1:s.nU)] += s.zU./((s.xU - s.x)[s.xU_bool]).*d[1:s.n][s.xU_bool] + r̄4./((s.xU - s.x)[s.xU_bool])

        # d .+= (s.Hu+Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m);zeros(s.nL+s.nU)]))\res

        res = -s.hu - s.Hu*d
        # println("d: $d")

        iter += 1
    end

    if norm(res,Inf) < s.opts.ϵ_iterative_refinement || norm(res,1) < res_norm
        s.d .= d
        println("iterative refinement success")
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
