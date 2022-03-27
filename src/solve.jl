function solve!(solver::Solver, x0)
    # solver
    s = solver

    # initialize
    initialize_primals!(s.variables, x0, s.indices)
    initialize_duals!(s.variables, s.indices)
    initialize_interior_point!(s.central_path, s.options)
    initialize_augmented_lagrangian!(s.penalty, s.dual, s.options)

    # initial step
    evaluate!(s)

    # initialize filter
    push!(s.filter, (s.max_constraint_violation, Inf))

    while tolerance(0.0, 0.0, s) > s.options.residual_tolerance
        while tolerance(s.central_path[1], s.penalty[1], s) > s.options.central_path_tolerance*s.central_path[1]
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
            evaluate!(s)

            s.residual_iteration += 1
            if s.residual_iteration > s.options.max_residual_iterations
                status(s)
                @error "max iterations"
                return
            end

        end

        if tolerance(0.0, 0.0, s) <= s.options.residual_tolerance && norm(s.problem.equality, Inf) <= s.options.equality_tolerance
            break
        else
            central_path_update!(s)
            augmented_lagrangian_update!(s)
            evaluate!(s)

            if s.residual_iteration == 0
                central_path_update!(s)
                augmented_lagrangian_update!(s)
                evaluate!(s)
            end
        end
    end  # outer while loop
    status(s)
end

function central_path_update!(s::Solver)
    central_path!(s)
    fraction_to_boundary!(s)

    s.outer_iteration += 1
    empty!(s.filter)
    push!(s.filter,(s.max_constraint_violation, Inf))
end

function status(s::Solver)
    if s.options.verbose
        println(crayon"red bold underline", "\nSolve Summary")
        println(crayon"reset", "   status: complete")
        println("   iteration ($(s.outer_iteration),$(s.residual_iteration)):")
        s.dimensions.variables < 5 &&  println("   x: $(s.variables)")
        println("   objective: $(s.methods.objective(s.variables))")
        println("   constraint_violation: $(s.constraint_violation), merit: $(s.merit)")
        println("   tolerance: $(tolerance(0.0, 0.0, s))")
        println("   norm(c,Inf): $(norm(s.c,Inf))")
        s.dimensions.equality > 0  && println("   norm(ce,Inf): $(norm(s.problem.equality,Inf))")
    end
end
