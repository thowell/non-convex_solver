function second_order_correction(s::Solver)
    status = false

    s.p = 1
    s.θ_soc = copy(s.θ)
    s.model.c_func!(s.c_soc,s.x⁺,s.model)
    if s.opts.nlp_scaling
        s.c_soc .= s.Dc*s.c_soc
    end
    s.c_soc .+= s.α*s.c

    search_direction_soc!(s)

    α_soc_max!(s)

    trial_step_soc!(s)

    while true
        if check_filter(s.θ⁺,s.φ⁺,s)
            # case 1
            if (s.θ <= s.θ_min && switching_condition(s))
                if armijo(s)
                    s.α = s.α_soc
                    status = true
                    println("second-order correction: success")
                    break
                end
            # case 2
            else#(s.θ > s.θ_min || !switching_condition(s))
                if sufficient_progress(s)
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

        if s.p == s.opts.p_max || s.θ⁺ > s.opts.κ_soc*s.θ_soc
            s.α = 0.5*s.α_max
            println("second-order correction: failure")
            break
        else
            s.p += 1

            s.model.c_func!(s.c,s.x⁺,s.model)
            s.c_soc .= s.α_soc*s.c_soc + (s.opts.nlp_scaling ? s.Dc*s.c : s.c)
            s.θ_soc = s.θ⁺

            search_direction_soc!(s)

            α_soc_max!(s)
            trial_step_soc!(s)
        end
    end

    return status
end

function trial_step_soc!(s::Solver)
    s.x⁺ .= s.x + s.α_soc*s.d_soc[s.idx.x]
    s.θ⁺ = θ(s.x⁺,s)
    s.φ⁺ = barrier(s.x⁺,s)
    return nothing
end

function α_soc_max!(s::Solver)
    s.α_soc = 1.0
    while !fraction_to_boundary_bnds(s.x,s.xL,s.xU,s.xL_bool,s.xU_bool,s.d_soc[s.idx.x],s.α_soc,s.τ)
        s.α_soc *= 0.5
    end
    return nothing
end

function search_direction_soc!(s::Solver)
    if s.opts.kkt_solve == :symmetric
        search_direction_soc_symmetric!(s)
    elseif s.opts.kkt_solve == :unreduced
        search_direction_soc_unreduced!(s)
    else
        error("KKT solve (soc) not implemented")
    end
    return nothing
end

function search_direction_soc_unreduced!(s::Solver)
    kkt_hessian_symmetric!(s)
    inertia_correction!(s)

    kkt_hessian_unreduced!(s)
    kkt_gradient_unreduced!(s)
    s.h[s.idx.y] = s.c_soc
    s.h[s.idx.y_al] += 1.0/s.ρ*(s.λ - s.y_al)

    s.d_soc .= lu(s.H + Diagonal(s.δ))\(-s.h)

    s.opts.iterative_refinement ? iterative_refinement(s.d_soc,s) : nothing

    return nothing
end

function search_direction_soc_symmetric!(s::Solver)
    kkt_hessian_symmetric!(s)
    kkt_gradient_symmetric!(s)
    s.h_sym[s.idx.y] = s.c_soc
    s.h_sym[s.idx.y_al] += 1.0/s.ρ*(s.λ - s.y_al)

    inertia_correction!(s)

    s.d_soc[s.idx.xy] = ma57_solve(s.LBL, -s.h_sym)
    s.d_soc[s.idx.zL] = -s.σL.*s.d_soc[s.idx.xL] - s.zL + s.μ./s.ΔxL
    s.d_soc[s.idx.zU] = s.σU.*s.d_soc[s.idx.xU] - s.zU + s.μ./s.ΔxU

    kkt_hessian_unreduced!(s)
    kkt_gradient_unreduced!(s)
    s.h[s.idx.y] = s.c_soc
    s.h[s.idx.y_al] += 1.0/s.ρ*(s.λ - s.y_al)

    s.opts.iterative_refinement ? iterative_refinement(s.d_soc,s) : nothing

    return nothing
end
