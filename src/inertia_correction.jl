function inertia_correction!(s::Solver; restoration=false,verbose=true)
    s.δw = 0.0
    s.δc = 0.0 #restoration ? 0. : s.μ #s.opts.δc*s.μ^s.opts.κc

    factorize_kkt!(s)

    verbose ? println("inertia-> n: $(s.inertia.n), m: $(s.inertia.m), z: $(s.inertia.z)") : nothing

    inertia(s) ? (return s.LBL) : nothing

    if s.inertia.z != 0
        verbose ? (println("$(s.inertia.z) zero eigen values")) : nothing
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
            verbose ? println("inertia (corrected)-> n: $(s.inertia.n), m: $(s.inertia.m), z: $(s.inertia.z)") : nothing
            break
        else
            if s.δw_last == 0
                s.δw = s.opts.κw⁺_*s.δw
            else
                s.δw = s.opts.κw⁺*s.δw
            end
        end

        if s.δw > s.opts.δw_max
            println("n: $(s.inertia.n), m: $(s.inertia.m), z: $(s.inertia.z)")
            println("s.δw: $(s.δw)")
            error("inertia correction failure")
        end
    end

    s.δw_last = s.δw

    return nothing
end

function factorize_kkt!(s::Solver)
    s.δ[s.idx.x] .= s.δw
    s.δ[s.idx.λ] .= -s.δc

    s.LBL = Ma57(s.H_sym + Diagonal(s.δ[s.idx.xλ]))
    ma57_factorize(s.LBL)

    s.inertia.m = s.LBL.info.num_negative_eigs
    s.inertia.n = s.LBL.info.rank - s.inertia.m
    s.inertia.z = s.model.n+s.model.m - s.LBL.info.rank

    return nothing
end

inertia(s::Solver) = (s.inertia.n == s.model.n
                        && s.inertia.m == s.model.m
                        && s.inertia.z == 0)
