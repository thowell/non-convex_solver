function second_order_correction(s::Solver)
    status = false

    s.p = 1
    θ_soc = s.θ
    s.c_soc .= s.α*s.c + s.c_func(s.x⁺)

    search_direction_soc!(s)

    α_soc_max!(s)

    s.x⁺ .= s.x + s.α_soc*s.d_soc[1:s.n]

    while true
        if check_filter(θ(s.x⁺,s),barrier(s.x⁺,s),s)
            # case 1
            if (s.θ <= s.θ_min && switching_condition(s))
                if armijo(s.x⁺,s)
                    s.α = s.α_soc
                    status = true
                    println("second order correction: success")
                    break
                end
            # case 2
            else#(s.θ > s.θ_min || !switching_condition(s))
                if sufficient_progress(s.x⁺,s)
                    s.α = s.α_soc
                    status = true
                    println("second order correction: success")
                    break
                end
            end
        else
            s.fail_cnt += 1
            s.α = 0.5*s.α_max
            println("second order correction: failure")
            break
        end

        if s.p == s.opts.p_max || θ(s.x⁺,s) > s.opts.κ_soc*θ_soc
            s.α = 0.5*s.α_max
            println("second order correction: failure")
            break
        else
            s.p += 1

            s.c_soc .= s.α_soc*s.c_soc + s.c_func(s.x⁺)
            θ_soc = θ(s.x⁺,s)

            search_direction_soc!(s)

            α_soc_max!(s)
        end
    end

    return status
end

function α_soc_max!(s::Solver)
    s.α_soc = 1.0
    while !fraction_to_boundary_bnds(s.x,s.xL,s.xU,s.xL_bool,s.xU_bool,s.d_soc[1:s.n],s.α_soc,s.τ)
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
        error("KKT solve not implemented")
    end
    return nothing
end

function search_direction_soc_symmetric!(s::Solver)
    s.h[s.n .+ (1:s.m)] = s.c_soc

    flag = inertia_correction_hsl!(s.H,s)

    LBL = Ma57(s.H + Diagonal([s.δw*ones(s.n);-s.δc*ones(s.m)]))
    ma57_factorize(LBL)
    s.d_soc[1:(s.n+s.m)] = ma57_solve(LBL, -s.h)
    # s.d_soc[1:(s.n+s.m)] = -s.H\s.h
    s.d_soc[(s.n+s.m) .+ (1:s.nL)] = -s.zL./((s.x - s.xL)[s.xL_bool]).*s.d[(1:s.n)[s.xL_bool]] - s.zL + s.μ./((s.x - s.xL)[s.xL_bool])
    s.d_soc[(s.n+s.m+s.nL) .+ (1:s.nU)] = s.zU./((s.xU - s.x)[s.xU_bool]).*s.d[(1:s.n)[s.xU_bool]] - s.zU + s.μ./((s.xU - s.x)[s.xU_bool])
    if flag
        iterative_refinement_soc(s.d_soc,s)
    end
    return nothing
end

function search_direction_soc_unreduced!(s::Solver)
    kkt_hessian_unreduced!(s)
    kkt_gradient_unreduced!(s)

    s.hu[s.n .+ (1:s.m)] = s.c_soc

    flag = inertia_correction_hsl!(s.H,s)
    δ = [s.δw*ones(s.n);-s.δc*ones(s.m);zeros(s.nL+s.nU)]

    s.d_soc .= -(s.Hu + Diagonal(δ))\s.hu

    if flag
        iterative_refinement(s.d_soc,δ,s)
    end

    s.δw = 0.
    s.δc = 0.

    return nothing
end
