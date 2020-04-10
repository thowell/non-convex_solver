function inertia_correction(s::Solver; restoration=false)
    s.δw = 0.0
    s.δc = restoration ? 0. : s.μ#s.opts.δc*s.μ^s.opts.κc

    LBL,n,m,z = compute_inertia(s)

    println("inertia-> n: $n, m: $m, z: $z")

    if n == s.model.n && z == 0 && m == s.model.m
        return LBL
    end

    if z != 0
        @warn "$z zero eigen values"
        s.δc = max(s.μ,s.opts.δc*s.μ^s.opts.κc)
    end

    if s.δw_last == 0.
        s.δw = max(s.δw,s.opts.δw0)
    else
        s.δw = max(s.δw,s.opts.δw_min,s.opts.κw⁻*s.δw_last)
    end

    while n != s.model.n || z != 0 || m != s.model.m

        LBL,n,m,z = compute_inertia(s)

        if n == s.model.n && z == 0 && m == s.model.m
            println("inertia (corrected)-> n: $n, m: $m, z: $z")
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

    return LBL
end

function compute_inertia(s::Solver)
    s.δ[s.idx.x] .= s.δw
    s.δ[s.idx.λ] .= -s.δc

    idx = [s.idx.x...,s.idx.λ...]

    LBL = Ma57(s.H_sym + Diagonal(s.δ[idx]))
    ma57_factorize(LBL)

    m = LBL.info.num_negative_eigs
    n = LBL.info.rank - m
    z = s.model.n+s.model.m - LBL.info.rank

    return LBL, n, m, z
end
