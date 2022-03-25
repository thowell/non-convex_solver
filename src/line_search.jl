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
        if check_filter(s.constraint_violation_candidate,s.merit⁺,s.filter)
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
    s.merit⁺ = barrier(s.candidate,s)
    return nothing
end

function minimum_step_size(d,constraint_violation,merit_gradient,min_constraint_violation,regularization,step_size_tolerance,constraint_violation_tolerance,merit_tolerance,exponent_constraint_violation,exponent_merit)
    if merit_gradient'*d < 0. && constraint_violation <= min_constraint_violation
        return step_size_tolerance*min(constraint_violation_tolerance,merit_tolerance*constraint_violation/(-merit_gradient'*d),regularization*(constraint_violation^exponent_constraint_violation)/(-merit_gradient'*d)^exponent_merit)
    elseif merit_gradient'*d < 0. && constraint_violation > min_constraint_violation
        return step_size_tolerance*min(constraint_violation_tolerance,merit_tolerance*constraint_violation/(-merit_gradient'*d))
    else
        return step_size_tolerance*constraint_violation_tolerance
    end
end

"""
    minimum_step_size!(s::Solver)

Compute the minimum step length (Eq. 23)
"""
function minimum_step_size!(s::Solver)
    s.minimum_step_size = minimum_step_size(s.dx,s.constraint_violation,s.merit_gradient,s.min_constraint_violation,s.options.regularization,s.options.step_size_tolerance,s.options.constraint_violation_tolerance,s.options.merit_tolerance,s.options.exponent_constraint_violation,s.options.exponent_merit)
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

switching_condition(merit_gradient,d,step_size,exponent_merit,regularization,constraint_violation,exponent_constraint_violation) = (merit_gradient'*d < 0. && step_size*(-merit_gradient'*d)^exponent_merit > regularization*constraint_violation^exponent_constraint_violation)
function switching_condition(s::Solver)
    return switching_condition(s.merit_gradient,s.dx,s.step_size,s.options.exponent_merit,s.options.regularization,s.constraint_violation,s.options.exponent_constraint_violation)
end

sufficient_progress(constraint_violation_candidate,constraint_violation,merit⁺,merit,constraint_violation_tolerance,merit_tolerance,machine_tolerance) = (constraint_violation_candidate - 10.0*machine_tolerance*abs(constraint_violation) <= (1-constraint_violation_tolerance)*constraint_violation || merit⁺ - 10.0*machine_tolerance*abs(merit) <= merit - merit_tolerance*constraint_violation)
function sufficient_progress(s::Solver)
    return sufficient_progress(s.constraint_violation_candidate,s.constraint_violation,s.merit⁺,s.merit,s.options.constraint_violation_tolerance,s.options.merit_tolerance,
        s.options.machine_tolerance)
end

armijo(merit⁺,merit,tolerance,step_size,merit_gradient,d,machine_tolerance) = (merit⁺ - merit - 10.0*machine_tolerance*abs(merit) <= tolerance*step_size*merit_gradient'*d)
armijo(s::Solver) = armijo(s.merit⁺,s.merit,s.options.armijo_tolerace,s.step_size,s.merit_gradient,s.dx,
    s.options.machine_tolerance)
