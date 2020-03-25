function search_direction!(s::Solver)
    ∇L(x) = s.∇f_func(x) + s.∇c_func(x)'*s.λ
    s.W .= ForwardDiff.jacobian(∇L,s.x)
    s.Σl[CartesianIndex.((1:s.n)[s.xl_bool],(1:s.n)[s.xl_bool])] .= s.zl./((s.x - s.xl)[s.xl_bool])
    s.Σu[CartesianIndex.((1:s.n)[s.xu_bool],(1:s.n)[s.xu_bool])] .= s.zu./((s.xu - s.x)[s.xu_bool])

    s.c .= s.c_func(s.x)
    s.A .= s.∇c_func(s.x)

    s.∇φ .= s.∇f_func(s.x)
    s.∇φ[s.xl_bool] .-= s.μ./(s.x - s.xl)[s.xl_bool]
    s.∇φ[s.xu_bool] .+= s.μ./(s.xu - s.x)[s.xu_bool]

    s.H[1:s.n,1:s.n] .= (s.W + s.Σl + s.Σu + s.δw*I)
    s.H[1:s.n,s.n .+ (1:s.m)] .= s.A'
    s.H[s.n .+ (1:s.m),1:s.n] .= s.A
    s.H[s.n .+ (1:s.m),s.n .+ (1:s.m)] .= -s.δc*Matrix(I,s.m,s.m)

    s.h[1:s.n] .= s.∇φ + s.A'*s.λ
    s.h[s.n .+ (1:s.m)] .= s.c

    s.d .= -s.H\s.h
    s.dzl .= -s.zl./((s.x - s.xl)[s.xl_bool]).*s.d[1:s.n][s.xl_bool] - s.zl + s.μ./((s.x - s.xl)[s.xl_bool])
    s.dzu .= s.zu./((s.xu - s.x)[s.xu_bool]).*s.d[1:s.n][s.xu_bool] - s.zu + s.μ./((s.xu - s.x)[s.xu_bool])
    return nothing
end

function α_min!(s::Solver)
    s.∇φ .= s.∇f_func(s.x)
    s.∇φ[s.xl_bool] .-= s.μ./(s.x - s.xl)[s.xl_bool]
    s.∇φ[s.xu_bool] .+= s.μ./(s.xu - s.x)[s.xu_bool]
    s.θ = θ(s.x,s)
    s.α_min = update_α_min(s.d[1:s.n],s.θ,s.∇φ,s.θ_min,s.opts.δ,s.opts.γα,s.opts.γθ,s.opts.γφ,s.opts.sθ,s.opts.sφ)

    # println("α_min: $(s.α_min)")
    return nothing
end

function α_max!(s::Solver)
    α_min!(s)

    s.α_max = 1.0
    while !fraction_to_boundary_bnds(s.x,s.xl,s.xu,s.xl_bool,s.xu_bool,s.d[1:s.n],s.α_max,s.τ)
        s.α_max *= 0.5
        println("α = $(s.α_max)")
        # if s.α_max < s.α_min
        #     error("α < α_min")
        # end
    end
    s.α = copy(s.α_max)

    s.αz = 1.0
    while !fraction_to_boundary(s.zl,s.dzl,s.αz,s.τ)
        s.αz *= 0.5
        println("αzl = $(s.αz)")
        # if s.αz < s.α_min
        #     error("αzl < α_min")
        # end
    end

    while !fraction_to_boundary(s.zu,s.dzu,s.αz,s.τ)
        s.αz *= 0.5
        println("αzu = $(s.αz)")
        # if s.αz < s.α_min
        #     error("αzu < α_min")
        # end
    end

    return nothing
end

function β_max!(s::Solver)

    s.β = 1.0
    while !fraction_to_boundary_bnds(s.x,s.xl,s.xu,s.xl_bool,s.xu_bool,s.d[1:s.n],s.β,s.τ)
        s.β *= 0.5
        println("β = $(s.β)")
        if s.β < 1.0e-32
            error("β < 1e-32 ")
        end
    end

    while !fraction_to_boundary(s.zl,s.dzl,s.β,s.τ)
        s.β *= 0.5
        println("β = $(s.β)")
        if s.β < 1.0e-32
            error("β < 1e-32 ")
        end
    end

    while !fraction_to_boundary(s.zu,s.dzu,s.β,s.τ)
        s.β *= 0.5
        println("β = $(s.β)")
        if s.β < 1.0e-32
            error("β < 1e-32 ")
        end
    end

    return nothing
end

function switching_condition(s::Solver)
    s.∇φ .= s.∇f_func(s.x)
    s.∇φ[s.xl_bool] .-= s.μ./(s.x - s.xl)[s.xl_bool]
    s.∇φ[s.xu_bool] .+= s.μ./(s.xu - s.x)[s.xu_bool]
    s.θ = θ(s.x,s)
    return switching_condition(s.∇φ,s.d[1:s.n],s.α,s.opts.sφ,s.opts.δ,s.θ,s.opts.sθ)
end

function sufficient_progress(x⁺,s::Solver)
    # println("θ⁺: $(θ(x⁺,s)), θ: $(θ(s.x,s))")
    # println("φ⁺: $(barrier(x⁺,s)), φ: $(barrier(s.x,s))")

    return sufficient_progress(θ(x⁺,s),θ(s.x,s),
        barrier(x⁺,s),barrier(s.x,s),s.opts.γθ,s.opts.γφ)
end

function armijo(x⁺,s::Solver)
    s.∇φ .= s.∇f_func(s.x)
    s.∇φ[s.xl_bool] .-= s.μ./(s.x - s.xl)[s.xl_bool]
    s.∇φ[s.xu_bool] .+= s.μ./(s.xu - s.x)[s.xu_bool]
    return armijo_condtion(barrier(x⁺,s),barrier(s.x,s),
        s.opts.ηφ,s.α,s.∇φ,s.d[1:s.n])
end

function line_search(s::Solver)
    α_max!(s)
    s.l = 0
    status = false
    θ0 = θ(s.x + s.α_max*s.d[1:s.n],s)
    φ0 = barrier(s.x + s.α_max*s.d[1:s.n],s)

    while s.α > s.α_min
        if check_filter(θ(s.x + s.α*s.d[1:s.n],s),barrier(s.x + s.α*s.d[1:s.n],s),s)
            # case 1
            if (θ(s.x,s) < s.θ_min && switching_condition(s))
                if armijo(s.x + s.α*s.d[1:s.n],s)
                    status = true
                    break
                end
            # case 2
            else
                if sufficient_progress(s.x + s.α*s.d[1:s.n],s)
                    status = true
                    break
                end
            end
        end

        if s.l > 0 || θ(s.x + s.α_max*s.d[1:s.n],s) < θ(s.x,s)
            s.α *= 0.5
        else
            if second_order_correction(s)
                status = true
                break
            end
        end
        # s.α *= 0.5

        s.l += 1
    end

    return status
end
