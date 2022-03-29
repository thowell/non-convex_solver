function line_search!(s::Solver)
    # reset 
    status = false
    s.line_search_iteration = 0 # line search interations

    # A-5.1 Initilize the line search
    # compute step_size, dual_step_size
    minimum_step_size!(s)  # update s.minimum_step_size
    maximum_step_size!(s)  # update s.maximum_step_size and s.step_size
    maximum_dual_step_size!(s)
    
    # A-5.2 Compute the new trial point
    candidate_step!(s)  # update s.candidate

    while s.step_size > s.minimum_step_size
        # A-5.3 Check acceptability to the filter
        if check_filter(s.constraint_violation_candidate, s.merit_candidate, s.filter)
            if s.line_search_iteration == 0
                s.failures = 0
            end

            # case 1
            if (s.constraint_violation <= s.min_constraint_violation && switching_condition(s.dx, s))
                if armijo(s)
                    status = true
                    break
                end

            # case 2
            else # (s.constraint_violation > s.min_constraint_violation || !switching_condition(s.dx, s))
                if sufficient_progress(s)
                    status = true
                    break
                end
            end
        end
        # TODO: the else here should go directly to A-5.10

        # A-5.5 Initialize the second-order correction
        if s.line_search_iteration > 0 || s.constraint_violation_candidate < s.constraint_violation
            # A-5.10 Choose new trail step size
            if s.line_search_iteration == 0
                s.failures += 1
            end
            s.step_size *= s.options.scaling_step_size
        else
            # A-5.6-9 Second order correction
            if second_order_correction(s)
                status = true
                break
            else
                s.failures += 1
            end
        end

        candidate_step!(s)
        s.line_search_iteration += 1
    end

    return status
end

"""
    candidate_step!(s::Solver)

Calculate the new candidate primal variables using the current step size and step.
Evaluate the constraint norm and the barrier objective at the new candidate.
"""
function candidate_step!(s::Solver)
    s.candidate .= s.x + s.step_size * s.dx
    s.constraint_violation_candidate = constraint_violation(s.candidate, s)
    s.merit_candidate = barrier(s.candidate, s)
    return nothing
end

"""
    minimum_step_size!(s::Solver)

Compute the minimum step length (Eq. 23)
"""
function minimum_step_size!(s::Solver)    
    d = s.dx
    θ = s.constraint_violation
    Mx = s.merit_gradient
    θmin = s.min_constraint_violation
    δ = s.options.regularization
    γα = s.options.step_size_tolerance
    γθ = s.options.constraint_violation_tolerance
    γM = s.options.merit_tolerance
    sθ = s.options.exponent_constraint_violation
    sM = s.options.exponent_merit
    
    if Mx' * d < 0.0 && θ <= θmin
        s.minimum_step_size = γα * min(γθ, γM * θ / (-Mx' * d), δ * (θ^sθ) / (-Mx' * d)^sM)
    elseif Mx' * d < 0.0 && θ > θmin
        s.minimum_step_size = γα * min(γθ, γM * θ / (-Mx' * d))
    else
        s.minimum_step_size = γα * γθ
    end

    return
end

"""
    maximum_step_size!(s::Solver)

Compute the maximum step length (Eq. 15)
"""
function maximum_step_size!(s::Solver)
    s.maximum_step_size = 1.0
    while !fraction_to_boundary_bounds(view(s.x,s.idx.xL),view(s.model.xL,s.idx.xL),s.dxL,s.maximum_step_size,s.fraction_to_boundary)
        s.maximum_step_size *= s.options.scaling_step_size
    end
    s.step_size = copy(s.maximum_step_size)

    return nothing
end

# TODO: these can all use the same function, since it's the same algorithm
function maximum_dual_step_size!(s::Solver)
    s.dual_step_size = 1.0
    while !fraction_to_boundary_bounds(s.zL,view(s.model.xL,s.idx.xL),s.dzL,s.dual_step_size,s.fraction_to_boundary)
        s.dual_step_size *= s.options.scaling_step_size
    end

    return nothing
end

#TODO: add reference
function switching_condition(d, s::Solver)
    Mx = s.merit_gradient
    α = s.step_size
    sM = s.options.exponent_merit
    δ = s.options.regularization
    θ = s.constraint_violation
    sθ = s.options.exponent_constraint_violation

    return (Mx' * d < 0.0 && α * (-Mx' * d)^sM > δ * θ^sθ)
end

# TODO: add reference
function sufficient_progress(s::Solver)
    θ_cand = s.constraint_violation_candidate
    θ = s.constraint_violation
    M_cand = s.merit_candidate
    M = s.merit
    γθ = s.options.constraint_violation_tolerance
    γM = s.options.merit_tolerance
    ϵ = s.options.machine_tolerance

    return (θ_cand - 10.0 * ϵ * abs(θ) <= (1.0 - γθ) * θ || M_cand - 10.0 * ϵ * abs(M) <= M - γM * θ)
end

# TODO: add reference
function armijo(s::Solver)
    M_cand = s.merit_candidate
    M = s.merit
    γa = s.options.armijo_tolerance
    α = s.step_size
    Mx = s.merit_gradient
    d = s.dx
    ϵ = s.options.machine_tolerance

    return (M_cand - M - 10.0 * ϵ * abs(M) <= γa * α * Mx' * d)
end


