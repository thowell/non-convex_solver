"""
    second_order_correction(s::Solver)

Implementation of A-5.7-9. Computes the second-order correction at the current trial point
`s.x⁺` and step size `s.α` and constraint values `s.c`. Corrected step is stored in `s.x⁺`.

Returns `true` if the correction was successful and `false` otherwise.
"""
function second_order_correction(s::Solver)
    status = false

    s.p = 1                                   # second-order correction iteration
    s.θ_soc = copy(s.θ)

    # Compute c_soc (Eq. 27)
    s.c_func!(s.c_soc, s.x⁺, s.model)   # evaluate the constraints at the current step
    if s.opts.nlp_scaling
        s.c_soc .= s.Dc*s.c_soc
    end
    s.c_soc .+= s.α*s.c

    # Compute the corrected search direction
    search_direction_soc!(s)

    # Calculate α_soc
    α_soc_max!(s)

    # Use α_soc to calculate the new step and the filter values
    trial_step_soc!(s)

    while true
        if check_filter(s.θ⁺,s.φ⁺,s.filter)  # A-5.7
            # case 1
            if (s.θ <= s.θ_min && switching_condition(s))  # A-5.8
                if armijo(s)
                    s.α = s.α_soc
                    status = true
                    s.opts.verbose && println("second-order correction: success")
                    break
                end
            # case 2
            else
                if sufficient_progress(s)
                    s.α = s.α_soc
                    status = true
                    s.opts.verbose && println("second-order correction: success")
                    break
                end
            end
        else
            s.α = 0.5*s.α_max
            s.opts.verbose && println("second-order correction: failure")
            break
        end

        if s.p == s.opts.p_max || s.θ⁺ > s.opts.κ_soc*s.θ_soc
            s.α = 0.5*s.α_max
            s.opts.verbose && println("second-order correction: failure")
            break
        else  # A-5.9 Next second-order correction
            s.p += 1

            s.c_func!(s.c,s.x⁺,s.model)
            s.c_soc .= s.α_soc*s.c_soc + (s.opts.nlp_scaling ? s.Dc*s.c : s.c)
            s.θ_soc = s.θ⁺

            search_direction_soc!(s)

            α_soc_max!(s)
            trial_step_soc!(s)
        end
    end

    return status
end

"""
    trial_step_soc!(s::Solver)

Calculate the new trial primal variables, as well as the constraint norm and barrier
objective values
"""
function trial_step_soc!(s::Solver)
    s.x⁺ .= s.x + s.α_soc*s.d_soc[s.idx.x]
    s.θ⁺ = θ(s.x⁺,s)
    s.φ⁺ = barrier(s.x⁺,s)
    return nothing
end

"""
    α_soc_max!(s::Solver)

Compute the approximate result of Eq. 28, the step size for the second order correction.
"""
function α_soc_max!(s::Solver)
    s.α_soc = 1.0
    while !fraction_to_boundary_bnds(s.x,s.xL,s.xU,s.xL_bool,s.xU_bool,s.d_soc[s.idx.x],s.α_soc,s.τ)
        s.α_soc *= 0.5  # QUESTION: is there a smarter way to find the approximate minimizer? Binary search? -maybe, no one does that
    end
    return nothing
end

function search_direction_soc!(s::Solver)
    if s.opts.kkt_solve == :symmetric
        search_direction_soc_symmetric!(s)
    elseif s.opts.kkt_solve == :fullspace
        search_direction_soc_fullspace!(s)
    else
        error("KKT solve (soc) not implemented")
    end
    return nothing
end

function search_direction_soc_fullspace!(s::Solver)
    kkt_hessian_symmetric!(s)
    inertia_correction!(s)

    kkt_hessian_fullspace!(s)
    kkt_gradient_fullspace!(s)
    s.h[s.idx.y] = s.c_soc
    s.h[s.idx.yA] += 1.0/s.ρ*(s.λ - s.yA)

    s.d_soc .= lu(s.H + Diagonal(s.δ))\(-s.h)

    s.opts.iterative_refinement ? iterative_refinement(s.d_soc,s) : nothing

    return nothing
end

"""
    search_direction_soc_symmetric!(s::Solver)

Compute the search direction `s.d_soc` by solving the reduced, symmetric KKT system (Eq. 26).
Can do iterative refinement based on the full fullspace KKT system if
    `s.opts.iterative_refinement = true`.

Here `s.d_soc` is the full corrected step after applying the second-order correction, i.e.
    x + α*dx + dx_soc
"""
function search_direction_soc_symmetric!(s::Solver)
    kkt_hessian_symmetric!(s)
    kkt_gradient_symmetric!(s)
    s.h_sym[s.idx.y] = s.c_soc
    s.h_sym[s.idx.yA] += 1.0/s.ρ*(s.λ - s.yA)

    inertia_correction!(s)

    # QUESTION: this seems to be re-computing the factorization? - no, solve does not recompute factorization. factorization happens in inertia correction
    s.dxy_soc .= ma57_solve(s.LBL, -s.h_sym)
    s.dzL_soc .= -s.σL.*s.dxL_soc - s.zL + s.μ./s.ΔxL  # Eq. 12
    s.dzU_soc .= s.σU.*s.dxU_soc - s.zU + s.μ./s.ΔxU   # Eq. 12

    if s.opts.iterative_refinement
        kkt_hessian_fullspace!(s)
        kkt_gradient_fullspace!(s)
        s.h[s.idx.y] = s.c_soc
        s.h[s.idx.yA] += 1.0/s.ρ*(s.λ - s.yA)
        iterative_refinement(s.d_soc,s)
    end

    return nothing
end
