function watch_dog!(s::Solver)
    # cache current iterate
    s.x_copy .= s.x
    s.y_copy .= s.y
    s.zL_copy .= s.zL
    s.zU_copy .= s.zU
    s.d_copy .= s.d

    # new trial point
    α_max!(s)
    αz_max!(s)
    α_max = min(s.α,s.αz)
    s.α = α_max
    s.αz = α_max
    trial_step!(s)
    accept_step!(s)

    # second new trial point
    eval_step!(s)
    search_direction!(s)
    α_max!(s)
    αz_max!(s)
    α_max = min(s.α,s.αz)
    s.α = α_max
    s.αz = α_max
    trial_step!(s)

    # check second new trial point with filter
    if check_filter(s.θ⁺,s.φ⁺,s)
        if s.l == 0
            s.fail_cnt = 0
        end

        # case 1
        if (s.θ <= s.θ_min && switching_condition(s))
            if armijo(s)
                @warn "watch dog -success-: armijo"
                return true
            end
        # case 2
        else #(s.θ > s.θ_min || !switching_condition(s))
            if sufficient_progress(s)
                @warn "watch dog -succces-: sufficient"
                return true
            end
        end
    else
        s.x .= s.x_copy
        s.y .= s.y_copy
        s.zL .= s.zL_copy
        s.zU .= s.zU_copy
        s.d .= s.d_copy
        eval_step!(s)
        @warn "watch dog -failure-:"
        return false
    end
end
