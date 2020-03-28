function inertia_correction!(s::Solver)
    n = -1
    m = -1
    z = -1

    try
        LDL = ldl(s.H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))
        n = sum(LDL.D .>= s.opts.ϵ_mach)
        m = sum(LDL.D .<= -s.opts.ϵ_mach)
        z = s.n+s.m - n - m
    catch

    end
    println("n: $n, m: $m, z: $z")
    # println("d: $(LDL.D)")
    if n == s.n && m == s.m && z == 0
        return false
    end

    if z != 0
        println("zeros eigen values")
        s.δc = s.opts.δc*s.μ^s.opts.κc
    end

    if s.δw_last == 0.
        s.δw = s.opts.δw0
    else
        s.δw = max(s.opts.δw_min,s.opts.κw⁻*s.δw_last)
    end

    while n != s.n || m != s.m || z != 0
        println("correcting interia ")
        # s.H[1:s.n,1:s.n] .= (s.W + s.Σl + s.Σu)
        # s.H[1:s.n,s.n .+ (1:s.m)] .= s.A'
        # s.H[s.n .+ (1:s.m),1:s.n] .= s.A
        # s.H[s.n .+ (1:s.m),s.n .+ (1:s.m)] .= -s.δc*Matrix(I,s.m,s.m)
        try
            LDL = ldl(s.H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))
            n = sum(LDL.D .>= s.opts.ϵ_mach)
            m = sum(LDL.D .<= -s.opts.ϵ_mach)
            z = s.n+s.m - n - m
        catch
            n = -1
            m = -1
            z = -1
        end

        if n == s.n || m == s.m || z == 0
            break
        else
            if s.δw_last == 0
                s.δw = s.opts.κw⁺_*s.δw
            else
                s.δw = s.opts.κw⁺*s.δw
            end
        end

        if s.δw > s.opts.δw_max
            error("inertia correction failure")
        end
    end

    s.δw_last = s.δw

    return true
end

function iterative_refinement(x,A,δ,b; max_iter=10,ϵ=1.0e-8,verbose=false)
    iter = 0
    res = b - A*x

    println("res: $res")
    while iter < max_iter && norm(res,Inf) > ϵ
        x .+= (A+Diagonal(δ))\res
        println("x: $x")

        res = b - A*x
        iter += 1
    end

    if verbose
        println("norm(res): $(norm(res))")
    end
    if iter < max_iter
        return true
    else
        false
    end
end
