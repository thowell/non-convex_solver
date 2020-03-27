function eval_Fμ(x,λ,zl,zu,s::Solver)
    s.c .= s.c_func(x)
    s.∇f .= s.∇f_func(x)
    s.A .= s.∇c_func(x)
    s.∇L .= s.∇f + s.A'*λ
    s.∇L[s.xl_bool] .-= zl
    s.∇L[s.xu_bool] .+= zu
    return eval_Fμ(x,λ,zl,zu,s.xl,s.xu,s.xl_bool,s.xu_bool,s.c,s.∇L,s.μ)
end

function barrier(x,s::Solver)
    return barrier(x,s.xl,s.xu,s.xl_bool,s.xu_bool,s.μ,s.f_func)
end

function θ(x,s::Solver)
    return norm(s.c_func(x),1)
end

function update_μ!(s::Solver)
    s.μ = update_μ(s.μ,s.opts.κμ,s.opts.θμ,s.opts.ϵ_tol)
    return nothing
end

function update_τ!(s::Solver)
    s.τ = update_τ(s.μ,s.opts.τ_min)
    return nothing
end

function reset_z!(s::Solver)
    for i = 1:s.nl
        s.zl[i] = reset_z(s.zl[i],s.x[s.xl_bool][i],s.μ,s.opts.κΣ)
    end

    for i = 1:s.nu
        s.zu[i] = reset_z(s.zu[i],s.x[s.xu_bool][i],s.μ,s.opts.κΣ)
    end
    return nothing
end

function update!(s::Solver)
    if s.update == :nominal
        s.x .= s.x + s.α*s.d[1:s.n]
    elseif s.update == :soc
        s.x .= s.x + s.α_soc*s.d_soc[1:s.n]
        s.update = :nominal # reset update
    else
        error("update error")
    end
    s.λ .= s.λ + s.α*s.d[s.n .+ (1:s.m)]
    s.zl .= s.zl + s.αz*s.dzl
    s.zu .= s.zu + s.αz*s.dzu

    reset_z!(s)
    return nothing
end

function solve!(s::Solver)
    println("--solve initiated--")
    θ0 = θ(s.x,s)
    φ0 = barrier(s.x,s)
    println("φ0: $φ0, θ0: $θ0")

    # initialize filter
    push!(s.filter,(s.θ_max,Inf))

    println("Eμ0: $(eval_Eμ(0.0,s))")
    while eval_Eμ(0.0,s) > s.opts.ϵ_tol
        while eval_Eμ(s.μ,s) > s.opts.κϵ*s.μ
            search_direction!(s)
            if !line_search(s)
                restoration!(s)
                augment_filter_restoration!(s.x,s)
            else
                augment_filter!(s)
                update!(s)
            end

            s.k += 1
            if s.k > s.opts.max_iter
                error("max iterations")
            end

            println("iteration (j,k): ($(s.j),$(s.k))")
            println("x: $(s.x)")
            println("Eμ: $(eval_Eμ(s.μ,s))")
            println("θjk: $(θ(s.x,s)), φjk: $(barrier(s.x,s))\n")
        end
        s.k = 0
        s.j += 1

        update_μ!(s)
        update_τ!(s)

        empty!(s.filter)
        push!(s.filter,(s.θ_max,Inf))
    end
end
