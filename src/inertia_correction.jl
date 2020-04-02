function inertia_correction_hsl!(H,s::Solver,restoration=false)
    LDL = Ma57(s.H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))
    ma57_factorize(LDL)

    m = LDL.info.num_negative_eigs
    n = LDL.info.rank - m
    z = s.n+s.m-LDL.info.rank

    println("n: $n, m: $m, z: $z")

    if n == s.n && z == 0 && m == s.m
        return false
    end

    if z != 0 && restoration==false
        println("$z zero eigen values")
        s.δc = s.opts.δc*s.μ^s.opts.κc
    end

    if s.δw_last == 0.
        s.δw = s.opts.δw0
    else
        s.δw = max(s.opts.δw_min,s.opts.κw⁻*s.δw_last)
    end

    while n != s.n || z != 0 || m != s.m
        LDL = Ma57(s.H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))
        ma57_factorize(LDL)

        m = LDL.info.num_negative_eigs
        n = LDL.info.rank - m
        z = s.n+s.m-LDL.info.rank

        if n == s.n && z == 0 && m == s.m
            @warn "δw: $(s.δw)"
            break
        else
            if s.δw_last == 0
                s.δw = s.opts.κw⁺_*s.δw
            else
                s.δw = s.opts.κw⁺*s.δw
            end
        end

        if s.δw > s.opts.δw_max
            println("n: $n, m: $m, z: $z")
            println("s.δw: $(s.δw)")
            error("inertia correction failure")
        end
    end

    s.δw_last = s.δw

    return true
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

# function iterative_refinement_phase1(x_,y,zL,zU,xL,xU,xL_bool,xU_bool,A,H,δ,b,n,nL,nU,m; max_iter=10,ϵ=1.0e-16,verbose=false)
#
#     x = copy(x_)
#     iter = 0
#     res = b - A*x
#
#     while iter < max_iter && norm(res,Inf) > ϵ
#
#         r12 = res[1:(n+m)]
#         r3 = res[(n+m) .+ (1:nL)]
#         r4 = res[(n+m+nL) .+ (1:nU)]
#
#         r12[(1:n)[xL_bool]] += r3./((y - xL)[xL_bool])
#         r12[(1:n)[xU_bool]] -= r4./((xU - y)[xU_bool])
#
#         # dxλ = -(H + Diagonal(δ[1:(n+m)]))\r12
#
#         LBL = Ma57(H + Diagonal(δ[1:(n+m)]))
#         ma57_factorize(LBL)
#         dxλ = ma57_solve(LBL, -r12)
#
#         println("hi")
#         dzL = -zL./((y - xL)[xL_bool]).*dxλ[(1:n)[xL_bool]] - r3./((y - xL)[xL_bool])
#         dzU = zU./((xU - y)[xU_bool]).*dxλ[(1:n)[xU_bool]] - r4./((xU - y)[xU_bool])
#
#
#         x .+= [dxλ;dzL;dzU]
#         # println("x: $x")
#
#         res = b - A*x
#         iter += 1
#     end
#
#     if norm(res,Inf) < ϵ
#         x_ .= x
#         println("iterative refinement success")
#         return true
#     else
#         println("iterative refinement failure: $(norm(res,Inf))")
#         println("δ: $(δ)")
#         false
#     end
# end
