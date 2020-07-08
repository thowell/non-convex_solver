function solve!(solver::NonConvexSolver)
    # phase 1 solver
    s = solver.s

    if s.opts.verbose
        println(crayon"bold underline red",
            "NON-CONVEX SOLVER")
        println(crayon"reset","Taylor Howell")
        println("Robotic Exploration Lab")
        println("Stanford University\n")
        println(crayon"bold underline black", "Problem Summary")
        print(crayon"reset")
        println("   num vars = $(s.model.n)")
        println("   num cons = $(s.model.m)")
        println()
    end

    # set up logger
    logger = SolverLogger(s.opts.verbose ? InnerLoop : Logging.Info)
    add_level!(logger, InnerLoop, print_color=:red)
    with_logger(logger) do

    # evaluate problem
    eval_step!(s)

    # initialize quasi-Newton
    s.qn.x_prev = copy(s.x)
    s.qn.∇f_prev = copy(get_∇f(s.model))
    s.qn.∇c_prev = copy(get_∇c(s.model))
    s.∇²L = copy(get_B(s.qn))
    s.model.mA > 0 && (view(s.∇²L,CartesianIndex.(s.idx.r,s.idx.r)) .+= s.ρ)

    # initialize filter
    push!(s.filter,(s.θ_max,Inf))

    # Print initial stats
    log_stats(s)
    print_level(InnerLoop)

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
                        @logmsg InnerLoop "small search direction"
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

                        @logmsg InnerLoop "infeasibility detected"
                        return
                    else
                        augment_filter!(s)
                        if s.opts.restoration
                            if !restoration!(solver.s̄,s)
                                @logmsg InnerLoop "restoration failed"
                                return
                            end
                        else
                            @logmsg InnerLoop "restoration mode turned off"
                            return
                        end
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
                @logmsg InnerLoop "max. iterations"
                return
            end

            log_stats(s)
            print_level(InnerLoop)

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

    end # logger
end

function barrier_update!(s::Solver)
    update_μ!(s)
    update_τ!(s)

    s.j += 1
    empty!(s.filter)
    push!(s.filter,(s.θ_max,Inf))
end

function log_stats(s)
    @logmsg InnerLoop :j value=s.j width=3
    @logmsg InnerLoop :k value=s.k width=3
    @logmsg InnerLoop :θ value=s.θ width=10
    @logmsg InnerLoop :φ value=s.φ width=10
    @logmsg InnerLoop :Eμ value=eval_Eμ(s.μ, s)
    @logmsg InnerLoop :f value=get_f(s,s.x) width=10
    @logmsg InnerLoop :μ value=s.μ width=10
    @logmsg InnerLoop :α value=s.α
end
