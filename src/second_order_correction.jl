function second_order_correction(s::Solver)
    status = false

    s.p = 1
    θ_soc = s.θ
    s.model.c_func!(s.c_soc,s.x⁺)
    s.c_soc .+= s.α*s.c

    search_direction_soc!(s)

    α_soc_max!(s)

    s.x⁺ .= s.x + s.α_soc*s.d_soc[s.idx.x]

    while true
        if check_filter(θ(s.x⁺,s),barrier(s.x⁺,s),s)
            # case 1
            if (s.θ <= s.θ_min && switching_condition(s))
                if armijo(s.x⁺,s)
                    s.α = s.α_soc
                    status = true
                    println("second-order correction: success")
                    break
                end
            # case 2
            else#(s.θ > s.θ_min || !switching_condition(s))
                if sufficient_progress(s.x⁺,s)
                    s.α = s.α_soc
                    status = true
                    println("second-order correction: success")
                    break
                end
            end
        else
            s.α = 0.5*s.α_max
            println("second-order correction: failure")
            break
        end

        if s.p == s.opts.p_max || θ(s.x⁺,s) > s.opts.κ_soc*θ_soc
            s.α = 0.5*s.α_max
            println("second-order correction: failure")
            break
        else
            s.p += 1

            s.model.c_func!(s.c,s.x⁺)
            s.c_soc .= s.α_soc*s.c_soc + s.c
            θ_soc = θ(s.x⁺,s)

            search_direction_soc!(s)

            α_soc_max!(s)
        end
    end

    return status
end

function α_soc_max!(s::Solver)
    s.α_soc = 1.0
    while !fraction_to_boundary_bnds(s.x,s.xL,s.xU,s.xL_bool,s.xU_bool,s.d_soc[s.idx.x],s.α_soc,s.τ)
        s.α_soc *= 0.5
    end
    return nothing
end

function search_direction_soc!(s::Solver)
    if s.opts.kkt_solve == :unreduced
        search_direction_soc_unreduced!(s)
    else
        error("KKT solve (soc) not implemented")
    end
    return nothing
end

function search_direction_soc_unreduced!(s::Solver)
    kkt_hessian_unreduced!(s)
    kkt_gradient_unreduced!(s)

    s.h[s.idx.λ] = s.c_soc

    flag = inertia_correction!(s)

    s.δ[s.idx.x] .= s.δw
    s.δ[s.idx.λ] .= -s.δc
    s.d_soc .= -(s.H + Diagonal(s.δ))\s.h

    if flag
        iterative_refinement(s.d_soc,s)
    end

    s.δw = 0.
    s.δc = 0.

    return nothing
end
