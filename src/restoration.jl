function check_kkt_error(s::Solver)
    Fμ = norm(eval_Fμ(s.x,s.λ,s.zl,s.zu,s),1)
    Fμ⁺ = norm(eval_Fμ(s.x + s.β*s.d[1:s.n], s.λ + s.β*s.d[s.n .+ (1:s.m)],
        s.zl + s.β*s.dzl, s.zu + s.β*s.dzu,s),1)

    println("Fμ: $(Fμ)")
    println("Fμ⁺: $(Fμ⁺)")
    println("kkt error: $((Fμ⁺ <= Fμ))")
    return (Fμ⁺ <= s.opts.κF*Fμ)
end

function restoration!(s::Solver)
    println("RESTORATION mode")
    status = false
    s.t = 0
    β_max!(s)

    while check_kkt_error(s)
        println("t: $(s.t)")
        if check_filter(θ(s.x + s.β*s.d[1:s.n],s),barrier(s.x + s.β*s.d[1:s.n],s),s::Solver)
            s.α = s.β
            s.αzl = s.β
            s.αzu = s.β
            println("KKT error reduction: success")
            status = true
            return
        else
            s.t += 1
            s.x .= s.x + s.β*s.d[1:s.n]
            s.λ .= s.λ + s.β*s.d[s.n .+ (1:s.m)]
            s.zl .= s.zl + s.β*s.dzl
            s.zu .= s.zu + s.β*s.dzu

            search_direction!(s)
            β_max!(s)
        end
    end

    if status
        return status
    else
        error("implement feasibility restoration")
    end
end

function set_DR!(s::Solver)
    set_DR(s.DR,s.x,s.n)
    return nothing
end

function init_n!(s::Solver)
    s.c .= s.c_func(s.x)
    for i = 1:s.m
        s.n_res[i] = init_n(s.c[i],s.μ_res,s.opts.ρ)
    end
    return nothing
end

function init_p!(s::Solver)
    s.c .= s.c_func(s.x)
    for i = 1:s.m
        s.p_res[i] = init_p(s.n_res[i],s.c[i])
    end
    return nothing
end
