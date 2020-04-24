function inertia_correction!(s::Solver; restoration=false)
    s.δw = 0.0
    s.δc = 0.0

    factorize_kkt!(s)

    s.opts.verbose ? println("inertia-> n: $(s.inertia.n), m: $(s.inertia.m), z: $(s.inertia.z)") : nothing

    inertia(s) ? (return nothing) : nothing

    if s.inertia.z != 0
        s.opts.verbose ? (println("$(s.inertia.z) zero eigen values")) : nothing
        s.δc = s.opts.δc*s.μ^s.opts.κc
    end

    if s.δw_last == 0.
        s.δw = max(s.δw,s.opts.δw0)
    else
        s.δw = max(s.δw,s.opts.δw_min,s.opts.κw⁻*s.δw_last)
    end

    while !inertia(s)

        factorize_kkt!(s)

        if inertia(s)
            s.opts.verbose ? println("inertia (corrected)-> n: $(s.inertia.n), m: $(s.inertia.m), z: $(s.inertia.z)") : nothing
            break
        else
            if s.δw_last == 0
                s.δw = s.opts.κw⁺_*s.δw
            else
                s.δw = s.opts.κw⁺*s.δw
            end
        end

        if s.δw > s.opts.δw_max
            if s.opts.verbose
                println("n: $(s.inertia.n), m: $(s.inertia.m), z: $(s.inertia.z)")
                println("s.δw: $(s.δw)")
            end
            error("inertia correction failure")
        end
    end

    s.δw_last = s.δw

    return nothing
end

function factorize_kkt!(s::Solver)
    s.δ[s.idx.x] .= s.δw
    s.δ[s.idx.y] .= -s.δc

    s.LBL = Ma57(s.H_sym + Diagonal(s.δ[s.idx.xy]))
    ma57_factorize(s.LBL)

    s.inertia.m = s.LBL.info.num_negative_eigs
    s.inertia.n = s.LBL.info.rank - s.inertia.m
    s.inertia.z = s.model.n+s.model.m - s.LBL.info.rank

    return nothing
end

inertia(s::Solver) = (s.inertia.n == s.model.n
                        && s.inertia.m == s.model.m
                        && s.inertia.z == 0)
