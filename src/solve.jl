function solve!(solver::NCSolver)
    # phase 1 solver
    s = solver.s

    # evaluate problem
    eval_step!(s)

    # initialize filter
    push!(s.filter,(s.θ_max,Inf))

    while eval_Eμ(0.0,s) > s.opts.ϵ_tol

        # Converge the interior point sub-problem
        while eval_Eμ(s.μ,s) > s.opts.κϵ*s.μ
            search_direction!(s)
    
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

            # s.opts.z_reset && reset_z!(s)

            # Calculate everything at the new trial point
            eval_step!(s)

            s.k += 1
            if s.k > s.opts.max_iter
                @error "max iterations"
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
