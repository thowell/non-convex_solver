function inertia_correction!(H,s::Solver)
    n = -1
    m = -1
    z = -1

    try
        LDL = ldl(H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))
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
        println("$z zero eigen values")
        s.δc = s.opts.δc*s.μ^s.opts.κc
    end

    if s.δw_last == 0.
        s.δw = s.opts.δw0
    else
        s.δw = max(s.opts.δw_min,s.opts.κw⁻*s.δw_last)
    end

    while n != s.n || m != s.m || z != 0
        println("correct interia ")
        try
            LDL = ldl(H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))
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

function iterative_refinement(x_,A,δ,b; max_iter=10,ϵ=1.0e-8,verbose=false)

    x = copy(x_)
    iter = 0
    res = b - A*x

    while iter < max_iter && norm(res,Inf) > ϵ
        x .+= (A+Diagonal(δ))\res
        println("x: $x")

        res = b - A*x
        iter += 1
    end

    if norm(res,Inf) < ϵ
        x_ .= x
        println("iterative refinement success")
        return true
    else
        println("iterative refinement failure")
        false
    end
end
