"""
    second_order_correction(s::Solver)

Implementation of A-5.7-9. Computes the second-order correction at the current trial point
`s.candidate` and step size `s.step_size` and constraint values `s.c`. Corrected step is stored in `s.candidate`.

Returns `true` if the correction was successful and `false` otherwise.
"""
function second_order_correction(s::Solver)
    status = false

    s.d_copy_2 .= s.d
    maximum_step_size = copy(s.maximum_step_size)
    
    s.p = 1
    s.constraint_violation_correction = copy(s.constraint_violation)

    # Compute c_soc (Eq. 27)
    eval_c!(s.model, s.candidate)
    get_c_scaled!(s.c_soc,s)

    s.c_soc .+= maximum_step_size*s.c

    # Compute the corrected search direction
    search_direction_soc!(s)

    # Calculate step_size_soc
    maximum_step_size!(s)

    # Use step_size_soc to calculate the new step and the filter values
    candidate_step!(s)

    while true
        if check_filter(s.constraint_violation_candidate,s.φ⁺,s.filter)  # A-5.7
            # case 1
            if (s.constraint_violation <= s.min_constraint_violation && switching_condition(s.∇φ,view(s.d_copy_2,s.idx.x),maximum_step_size,s.options.exponent_merit,s.options.regularization,s.constraint_violation,s.options.sconstraint_violation))  # A-5.8
                if armijo(s)
                    status = true
                    break
                end
            # case 2
            else
                if sufficient_progress(s)
                    status = true
                    break
                end
            end
        else
            s.step_size = 0.5*maximum_step_size
            break
        end

        if s.p == s.options.max_second_order_correction || s.constraint_violation_candidate > s.options.soc_tolerance*s.constraint_violation_correction
            s.step_size = 0.5*maximum_step_size
            break
        else  # A-5.9 Next second-order correction
            s.p += 1

            eval_c!(s.model,s.candidate)
            get_c_scaled!(s.c,s)

            s.c_soc .= s.step_size*s.c_soc + s.c
            s.constraint_violation_correction = s.constraint_violation_candidate

            search_direction_soc!(s)

            maximum_step_size!(s)
            candidate_step!(s)
        end
    end

    s.d .= s.d_copy_2
    return status
end

function search_direction_soc!(s::Solver)
    s.hy .= s.c_soc
    search_direction!(s)
end
