function line_search(s::Solver)
    # compute α, αz
    α_min!(s)
    α_max!(s)
    αz_max!(s)

    # trial step
    s.x⁺ .= s.x + s.α*s.dx

    s.l = 0
    status = false
    while s.α > s.α_min
        if check_filter(θ(s.x⁺,s),barrier(s.x⁺,s),s)
            # case 1
            if (s.θ <= s.θ_min && switching_condition(s))
                if armijo(s.x⁺,s)
                    status = true
                    break
                end
            # case 2
            else #(s.θ > s.θ_min || !switching_condition(s))
                if sufficient_progress(s.x⁺,s)
                    status = true
                    break
                end
            end
        end
        if s.l > 0 || θ(s.x⁺,s) < s.θ || s.restoration == true
            s.α *= 0.5
        else
            # second order correction
            if second_order_correction(s)
                println("--success confirm")
                status = true
                break
            else
                println("--failure confirm")
            end
        end

        s.x⁺ .= s.x + s.α*s.dx

        s.l += 1
    end

    return status
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

function α_min!(s::Solver)
    s.α_min = α_min(s.dx,s.θ,s.∇φ,s.θ_min,s.opts.δ,s.opts.γα,s.opts.γθ,s.opts.γφ,s.opts.sθ,s.opts.sφ)
    return nothing
end

function α_max!(s::Solver)
    s.α_max = 1.0
    while !fraction_to_boundary_bnds(s.x,s.xL,s.xU,s.xL_bool,s.xU_bool,s.dx,s.α_max,s.τ)
        s.α_max *= 0.5
    end
    s.α = copy(s.α_max)

    return nothing
end

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

# function β_max!(s::Solver)
#
#     s.β = 1.0
#     while !fraction_to_boundary_bnds(s.x,s.xL,s.xU,s.xL_bool,s.xU_bool,s.dx,s.β,s.τ)
#         s.β *= 0.5
#         println("β = $(s.β)")
#         if s.β < 1.0e-32
#             error("β < 1e-32 ")
#         end
#     end
#
#     while !fraction_to_boundary(s.zL,s.dzL,s.β,s.τ)
#         s.β *= 0.5
#         println("β = $(s.β)")
#         if s.β < 1.0e-32
#             error("β < 1e-32 ")
#         end
#     end
#
#     while !fraction_to_boundary(s.zU,s.dzU,s.β,s.τ)
#         s.β *= 0.5
#         println("β = $(s.β)")
#         if s.β < 1.0e-32
#             error("β < 1e-32 ")
#         end
#     end
#
#     return nothing
# end

switching_condition(∇φ,d,α,sφ,δ,θ,sθ) = (∇φ'*d < 0. && α*(-∇φ'*d)^sφ > δ*θ^sθ)
function switching_condition(s::Solver)
    return switching_condition(s.∇φ,s.dx,s.α,s.opts.sφ,s.opts.δ,s.θ,s.opts.sθ)
end

sufficient_progress(θ⁺,θ,φ⁺,φ,γθ,γφ) = (θ⁺ <= (1-γθ)*θ || φ⁺ <= φ - γφ*θ)
function sufficient_progress(x⁺,s::Solver)
    return sufficient_progress(θ(x⁺,s),s.θ,barrier(x⁺,s),s.φ,s.opts.γθ,s.opts.γφ)
end

armijo(φ⁺,φ,η,α,∇φ,d) = (φ⁺ <= φ + η*α*∇φ'*d)
armijo(x⁺,s::Solver) = armijo(barrier(x⁺,s),s.φ,s.opts.ηφ,s.α,s.∇φ,s.dx)
