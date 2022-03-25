function solve!(solver::Solver)
    # initiali step
    step!(s)

    # initialize filter
    push!(s.filter,(s.max_constraint_violation,Inf))

    while tolerance(0.0,s) > s.options.residual_tolerance
        while tolerance(s.central_path,s) > s.options.central_path_tolerance*s.central_path
            search_direction!(s)
    
            if !line_search!(s)
                if s.constraint_violation < s.options.residual_tolerance
                    status(s)
                    @error "infeasibility detected"
                    return
                else
                    augment_filter!(s)
                    status(s)
                    return
                end
            else  # successful line search
                augment_filter!(s)
                accept_step!(s)
            end

            # Calculate everything at the new trial point
            step!(s)

            s.k += 1
            if s.k > s.options.max_residual_iterations
                status(s)
                @error "max iterations"
                return
            end

        end

        if tolerance(0.0,s) <= s.options.residual_tolerance && norm(s.xr,Inf) <= s.options.equality_tolerance
            break
        else
            barrier_update!(s)
            augmented_lagrangian_update!(s)
            step!(s)

            if s.k == 0
                barrier_update!(s)
                augmented_lagrangian_update!(s)
                step!(s)
            end
        end
    end  # outer while loop
    status(s)
end

function barrier_update!(s::Solver)
    central_path!(s)
    fraction_to_boundary!(s)

    s.j += 1
    empty!(s.filter)
    push!(s.filter,(s.max_constraint_violation,Inf))
end

function status(s::Solver)
    if s.options.verbose
        println(crayon"red bold underline", "\nSolve Summary")
        println(crayon"reset", "   status: complete")
        println("   iteration ($(s.j),$(s.k)):")
        s.model.n < 5 &&  println("   x: $(s.x)")
        println("   objective: $(get_f(s,s.x))")
        println("   constraint_violation: $(s.constraint_violation), φ: $(s.φ)")
        println("   tolerance: $(tolerance(0.0,s))")
        println("   norm(c,Inf): $(norm(s.c,Inf))")
        s.model.mA > 0  && println("   norm(r,Inf): $(norm(s.xr,Inf))")
    end
end
