"""
    second_order_correction(s::Solver)

Implementation of A-5.7-9. Computes the second-order correction at the current trial point
`s.x⁺` and step size `s.α` and constraint values `s.c`. Corrected step is stored in `s.x⁺`.

Returns `true` if the correction was successful and `false` otherwise.
"""
function second_order_correction(s::Solver)
    status = false

    s.d_copy_2 .= s.d
    α_max = copy(s.α_max)

    s.p = 1
    s.θ_soc = copy(s.θ)

    # Compute c_soc (Eq. 27)
    eval_c!(s.model,s.x⁺)
    get_c_scaled!(s.c_soc,s)

    s.c_soc .+= α_max*s.c

    # Compute the corrected search direction
    search_direction_soc!(s)

    # Calculate α_soc
    α_max!(s)

    # Use α_soc to calculate the new step and the filter values
    trial_step!(s)

    while true
        if check_filter(s.θ⁺,s.φ⁺,s.filter)  # A-5.7
            # case 1
            if (s.θ <= s.θ_min && switching_condition(s.∇φ,view(s.d_copy_2,s.idx.x),α_max,s.opts.sφ,s.opts.δ,s.θ,s.opts.sθ))  # A-5.8
                if armijo(s)
                    status = true
                    # @logmsg InnerLoop "soc: +"
                    break
                end
            # case 2
            else
                if sufficient_progress(s)
                    status = true
                    # @logmsg InnerLoop "soc: +"
                    break
                end
            end
        else
            s.α = 0.5*α_max
            # @logmsg InnerLoop "soc: x"
            break
        end

        if s.p == s.opts.p_max || s.θ⁺ > s.opts.κ_soc*s.θ_soc
            s.α = 0.5*α_max
            # @logmsg InnerLoop "soc: x"
            break
        else  # A-5.9 Next second-order correction
            s.p += 1

            eval_c!(s.model,s.x⁺)
            get_c_scaled!(s.c,s)

            s.c_soc .= s.α*s.c_soc + s.c
            s.θ_soc = s.θ⁺

            search_direction_soc!(s)

            α_max!(s)
            trial_step!(s)
        end
    end

    s.d .= s.d_copy_2
    return status
end

function search_direction_soc!(s::Solver)
    s.hy .= s.c_soc
    search_direction!(s)
end
