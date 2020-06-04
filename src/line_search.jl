function line_search(s::Solver)
    # A-5.1 Initilize the line search
    # compute α, αz
    α_min!(s)  # update s.α_min
    α_max!(s)  # update s.α_max and s.α
    αz_max!(s)
    s.l = 0   # line search interations
    status = false

    # A-5.2 Compute the new trial point
    trial_step!(s)  # update s.x⁺

    while s.α > s.α_min
        # A-5.3 Check acceptability to the filter
        if check_filter(s.θ⁺,s.φ⁺,s.filter)
            if s.l == 0
                s.fail_cnt = 0
            end

            # case 1
            if (s.θ <= s.θ_min && switching_condition(s))
                if armijo(s)
                    status = true
                    break
                end
            # case 2
            else #(s.θ > s.θ_min || !switching_condition(s))
                if sufficient_progress(s)
                    status = true
                    break
                end
            end
        end
        # TODO: the else here should go directly to A-5.10

        # A-5.5 Initialize the second-order correction
        if s.l > 0 || s.θ⁺ < s.θ || s.restoration == true
            # A-5.10 Choose new trail step size
            if s.l == 0
                s.fail_cnt += 1
            end
            s.α *= 0.5
        else
            # A-5.6-9 Second order correction
            if second_order_correction(s)
                status = true
                break
            else
                s.fail_cnt += 1
            end
        end

        # accelerating heuristics
        if s.fail_cnt == s.opts.max_fail_cnt
            s.fail_cnt = 0
            if s.θ_max > 0.1*s.θ⁺
                s.θ_max *= 0.1
                empty!(s.filter)
                push!(s.filter,(s.θ_max,Inf))
                s.opts.verbose && @warn "acceleration heuristic: resetting filter, reducing θ_max"
            else
                # @warn "WATCH DOG : UNTESTED"
                # if watch_dog!(s)
                #     s.opts.verbose && @warn "acceleration heuristic: watch dog -success"
                #     status = true
                #     break
                # end
            end
        end

        trial_step!(s)

        s.l += 1
    end
    return status
end

"""
    trial_step!(s::Solver)

Calculate the new candidate primal variables using the current step size and step.
Evaluate the constraint norm and the barrier objective at the new candidate.
"""
function trial_step!(s::Solver)
    s.x⁺ .= s.x + s.α*s.dx
    s.θ⁺ = θ(s.x⁺,s)
    s.φ⁺ = barrier(s.x⁺,s)
    return nothing
end

function α_min(d,θ,∇φ,θ_min,δ,γα,γθ,γφ,sθ,sφ)
    if ∇φ'*d < 0. && θ <= θ_min
        α_min = γα*min(γθ,γφ*θ/(-∇φ'*d),δ*(θ^sθ)/(-∇φ'*d)^sφ)
    elseif ∇φ'*d < 0. && θ > θ_min
        α_min = γα*min(γθ,γφ*θ/(-∇φ'*d))
    else
        α_min = γα*γθ
    end
    return α_min
end

"""
    α_min!(s::Solver)

Compute the minimum step length (Eq. 23)
"""
function α_min!(s::Solver)
    s.α_min = α_min(s.dx,s.θ,s.∇φ,s.θ_min,s.opts.δ,s.opts.γα,s.opts.γθ,s.opts.γφ,s.opts.sθ,s.opts.sφ)
    return nothing
end

"""
    α_max!(s::Solver)

Compute the maximum step length (Eq. 15)
"""
function α_max!(s::Solver)
    s.α_max = 1.0
    while !fraction_to_boundary_bnds(view(s.x,s.idx.xL),view(s.model.xL,s.idx.xL),view(s.x,s.idx.xU),view(s.model.xU,s.idx.xU),s.dxL,s.dxU,s.α_max,s.τ)
        s.α_max *= 0.5
    end
    s.α = copy(s.α_max)

    return nothing
end

# TODO: these can all use the same function, since it's the same algorithm
function αz_max!(s::Solver)
    s.αz = 1.0
    while !fraction_to_boundary(s.zL,s.dzL,s.αz,s.τ)
        s.αz *= 0.5
    end

    while !fraction_to_boundary(s.zU,s.dzU,s.αz,s.τ)
        s.αz *= 0.5
    end

    return nothing
end

switching_condition(∇φ,d,α,sφ,δ,θ,sθ) = (∇φ'*d < 0. && α*(-∇φ'*d)^sφ > δ*θ^sθ)
function switching_condition(s::Solver)
    return switching_condition(s.∇φ,s.dx,s.α,s.opts.sφ,s.opts.δ,s.θ,s.opts.sθ)
end

sufficient_progress(θ⁺,θ,φ⁺,φ,γθ,γφ,ϵ_mach) = (θ⁺ - 10.0*ϵ_mach*abs(θ) <= (1-γθ)*θ || φ⁺ - 10.0*ϵ_mach*abs(φ) <= φ - γφ*θ)
function sufficient_progress(s::Solver)
    return sufficient_progress(s.θ⁺,s.θ,s.φ⁺,s.φ,s.opts.γθ,s.opts.γφ,
        s.opts.ϵ_mach)
end

armijo(φ⁺,φ,η,α,∇φ,d,ϵ_mach) = (φ⁺ - φ - 10.0*ϵ_mach*abs(φ) <= η*α*∇φ'*d)
armijo(s::Solver) = armijo(s.φ⁺,s.φ,s.opts.ηφ,s.α,s.∇φ,s.dx,
    s.opts.ϵ_mach)
