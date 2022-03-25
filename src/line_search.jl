function line_search!(s::Solver)
    # A-5.1 Initilize the line search
    # compute step_size, dual_step_size
    minimum_step_size!(s)  # update s.minimum_step_size
    maximum_step_size!(s)  # update s.maximum_step_size and s.step_size
    maximum_dual_step_size!(s)
    
    s.l = 0   # line search interations
    status = false

    # A-5.2 Compute the new trial point
    candidate_step!(s)  # update s.candidate

    while s.step_size > s.minimum_step_size
        # A-5.3 Check acceptability to the filter
        if check_filter(s.constraint_violation_candidate,s.φ⁺,s.filter)
            if s.l == 0
                s.fail_cnt = 0
            end

            # case 1
            if (s.constraint_violation <= s.min_constraint_violation && switching_condition(s))
                if armijo(s)
                    status = true
                    break
                end
            # case 2
            else #(s.constraint_violation > s.min_constraint_violation || !switching_condition(s))
                if sufficient_progress(s)
                    status = true
                    break
                end
            end
        end
        # TODO: the else here should go directly to A-5.10

        # A-5.5 Initialize the second-order correction
        if s.l > 0 || s.constraint_violation_candidate < s.constraint_violation #|| true
            # A-5.10 Choose new trail step size
            if s.l == 0
                s.fail_cnt += 1
            end
            s.step_size *= 0.5
        else
            # A-5.6-9 Second order correction
            if second_order_correction(s)
                status = true
                break
            else
                s.fail_cnt += 1
            end
        end

        candidate_step!(s)

        s.l += 1
    end
    return status
end

"""
    candidate_step!(s::Solver)

Calculate the new candidate primal variables using the current step size and step.
Evaluate the constraint norm and the barrier objective at the new candidate.
"""
function candidate_step!(s::Solver)
    s.candidate .= s.x + s.step_size*s.dx
    s.constraint_violation_candidate = constraint_violation(s.candidate,s)
    s.φ⁺ = barrier(s.candidate,s)
    return nothing
end

function minimum_step_size(d,constraint_violation,∇φ,min_constraint_violation,regularization,step_size_tolerance,constraint_violation_tolerance,γφ,sθ,sφ)
    if ∇φ'*d < 0. && constraint_violation <= min_constraint_violation
        return step_size_tolerance*min(constraint_violation_tolerance,γφ*constraint_violation/(-∇φ'*d),regularization*(constraint_violation^sθ)/(-∇φ'*d)^sφ)
    elseif ∇φ'*d < 0. && constraint_violation > min_constraint_violation
        return step_size_tolerance*min(constraint_violation_tolerance,γφ*constraint_violation/(-∇φ'*d))
    else
        return step_size_tolerance*constraint_violation_tolerance
    end
end

"""
    minimum_step_size!(s::Solver)

Compute the minimum step length (Eq. 23)
"""
function minimum_step_size!(s::Solver)
    s.minimum_step_size = minimum_step_size(s.dx,s.constraint_violation,s.∇φ,s.min_constraint_violation,s.options.regularization,s.options.step_size_tolerance,s.options.constraint_violation_tolerance,s.options.γφ,s.options.sθ,s.options.sφ)
    return nothing
end

"""
    maximum_step_size!(s::Solver)

Compute the maximum step length (Eq. 15)
"""
function maximum_step_size!(s::Solver)
    s.maximum_step_size = 1.0
    while !fraction_to_boundary_bounds(view(s.x,s.idx.xL),view(s.model.xL,s.idx.xL),view(s.x,s.idx.xU),view(s.model.xU,s.idx.xU),s.dxL,s.dxU,s.maximum_step_size,s.fraction_to_boundary)
        s.maximum_step_size *= 0.5
    end
    s.step_size = copy(s.maximum_step_size)

    return nothing
end

# TODO: these can all use the same function, since it's the same algorithm
function maximum_dual_step_size!(s::Solver)
    s.dual_step_size = 1.0
    while !fraction_to_boundary(s.zL,s.dzL,s.dual_step_size,s.fraction_to_boundary)
        s.dual_step_size *= 0.5
    end

    while !fraction_to_boundary(s.zU,s.dzU,s.dual_step_size,s.fraction_to_boundary)
        s.dual_step_size *= 0.5
    end

    return nothing
end

switching_condition(∇φ,d,step_size,sφ,regularization,constraint_violation,sθ) = (∇φ'*d < 0. && step_size*(-∇φ'*d)^sφ > regularization*constraint_violation^sθ)
function switching_condition(s::Solver)
    return switching_condition(s.∇φ,s.dx,s.step_size,s.options.sφ,s.options.regularization,s.constraint_violation,s.options.sθ)
end

sufficient_progress(constraint_violation_candidate,constraint_violation,φ⁺,φ,constraint_violation_tolerance,γφ,machine_tolerance) = (constraint_violation_candidate - 10.0*machine_tolerance*abs(constraint_violation) <= (1-constraint_violation_tolerance)*constraint_violation || φ⁺ - 10.0*machine_tolerance*abs(φ) <= φ - γφ*constraint_violation)
function sufficient_progress(s::Solver)
    return sufficient_progress(s.constraint_violation_candidate,s.constraint_violation,s.φ⁺,s.φ,s.options.constraint_violation_tolerance,s.options.γφ,
        s.options.machine_tolerance)
end

armijo(φ⁺,φ,η,step_size,∇φ,d,machine_tolerance) = (φ⁺ - φ - 10.0*machine_tolerance*abs(φ) <= η*step_size*∇φ'*d)
armijo(s::Solver) = armijo(s.φ⁺,s.φ,s.options.ηφ,s.step_size,s.∇φ,s.dx,
    s.options.machine_tolerance)
