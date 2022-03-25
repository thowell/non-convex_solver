function solve!(solver::NCSolver)
    # phase 1 solver
    s = solver.s

    # evaluate problem
    eval_step!(s)

    s.∇²L = copy(get_B(s.qn))
    s.model.mA > 0 && (view(s.∇²L,CartesianIndex.(s.idx.r,s.idx.r)) .+= s.ρ)

    # initialize filter
    push!(s.filter,(s.θ_max,Inf))

    while eval_Eμ(0.0,s) > s.opts.ϵ_tol

        # Converge the interior point sub-problem
        while eval_Eμ(s.μ,s) > s.opts.κϵ*s.μ
            s.opts.relax_bnds && relax_bounds!(s)

            # solve for the search direction and check if it's small
            if search_direction!(s)
                s.small_search_direction_cnt += 1
                if s.small_search_direction_cnt == s.opts.small_search_direction_max
                    s.small_search_direction_cnt = 0
                    if s.μ < 0.1*s.opts.ϵ_tol
                        return
                    else
                        break
                    end
                end
                α_max!(s)
                αz_max!(s)
                augment_filter!(s)
                accept_step!(s)
            else
                s.small_search_direction_cnt = 0

                # Perform line search and check if it fails
                if !line_search(s)
                    if s.θ < s.opts.ϵ_tol && s.opts.quasi_newton == :none
                        @error "infeasibility detected"
                        return
                    else
                        augment_filter!(s)
                        return
                    end
                else  # successful line search
                    augment_filter!(s)
                    accept_step!(s)
                end
            end

            s.opts.z_reset && reset_z!(s)

            # Calculate everything at the new trial point
            eval_step!(s)
            update_quasi_newton!(s)

            s.k += 1
            if s.k > s.opts.max_iter
                @error "max iterations"
                # @logmsg InnerLoop "max. iterations"
                return
            end

        end  # inner while loop


        if eval_Eμ(0.0,s) <= s.opts.ϵ_tol && norm(s.xr,Inf) <= s.opts.ϵ_al_tol
            break
        else
            barrier_update!(s)
            augmented_lagrangian_update!(s)
            eval_step!(s)

            if s.k == 0
                barrier_update!(s)
                augmented_lagrangian_update!(s)
                eval_step!(s)
            end
        end
    end  # outer while loop

    if s.opts.verbose
        println(crayon"red bold underline", "\nSolve Summary")
        println(crayon"reset", "   status: complete")
        println("   iteration ($(s.j),$(s.k)):")
        s.model.n < 5 &&  println("   x: $(s.x)")
        println("   f: $(get_f(s,s.x))")
        println("   θ: $(s.θ), φ: $(s.φ)")
        println("   E0: $(eval_Eμ(0.0,s))")
        println("   norm(c,Inf): $(norm(s.c,Inf))")
        s.model.mA > 0  && println("   norm(r,Inf): $(norm(s.xr,Inf))")
    end
end

function barrier_update!(s::Solver)
    update_μ!(s)
    update_τ!(s)

    s.j += 1
    empty!(s.filter)
    push!(s.filter,(s.θ_max,Inf))
end
