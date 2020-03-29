function second_order_correction(s::Solver)
    s.p = 1
    θ_soc = s.θ
    c_soc = s.α*s.c + s.c_func(s.x⁺)
    s.h[s.n .+ (1:s.m)] .= c_soc
    s.d_soc[1:(s.n+s.m)] = -s.H\s.h

    # α_soc_max
    s.α_soc = 1.0
    while !fraction_to_boundary_bnds(s.x,s.xl,s.xu,s.xl_bool,s.xu_bool,s.d_soc[1:s.n],s.α_soc,s.τ)
        s.α_soc *= 0.5
    end

    s.x⁺ .= s.x + s.α_soc*s.d_soc[1:s.n]

    while true
        if check_filter(θ(s.x⁺,s),barrier(s.x⁺,s),s)
            # case 1
            if (s.θ <= s.θ_min && switching_condition(s))
                if armijo(s.x⁺,s)
                    s.update = :soc
                    s.α = s.α_soc
                    break
                end
            # case 2
            else#(s.θ > s.θ_min || !switching_condition(s))
                if sufficient_progress(s.x⁺,s)
                    s.update = :soc
                    s.α = s.α_soc
                    break
                end
            end
        else
            s.α = 0.5*s.α_max
            break
        end

        if s.p == s.opts.p_max || θ(s.x⁺,s) > s.opts.κ_soc*θ_soc
            s.α = 0.5*s.α_max
            break
        else
            s.p += 1

            s.c_soc .= s.α_soc*s.c_soc + s.c_func(s.x⁺)
            θ_soc = θ(s.x⁺,s)

            s.h[s.n .+ (1:s.m)] .= c_soc
            s.d_soc .= -s.H\s.h

            s.α_soc = 1.0
            while !fraction_to_boundary_bnds(s.x,s.xl,s.xu,s.xl_bool,s.xu_bool,s.d_soc[1:s.n],s.α_soc,s.τ)
                s.α_soc *= 0.5
            end
        end
    end

    if s.update == :soc
        println("second order correction: success")
        return true
    else
        println("second order correction: failure")
        return false
    end
end
