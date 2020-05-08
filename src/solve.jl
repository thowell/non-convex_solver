using Crayons
function solve!(solver::InteriorPointSolver)
    # println("<augmented-Lagrangian interior-point solve>\n")


    s = solver.s

    # Problem summar

    # evaluate problem
    eval_step!(s)

    # initialize filter
    push!(s.filter,(s.θ_max,Inf))

    # Print initial stats

    while eval_Eμ(0.0,s) > s.opts.ϵ_tol

        # Converge the interior point sub-problem
        # s.k > 0 && print_header(logger, InnerLoop)
        while eval_Eμ(s.μ,s) > s.opts.κϵ*s.μ
            s.opts.relax_bnds && relax_bnds!(s)

            # solve for the search direction and check if it's small
            if search_direction!(s)
                s.small_search_direction_cnt += 1
                if s.small_search_direction_cnt == s.opts.small_search_direction_max
                    s.small_search_direction_cnt = 0
                    if s.μ < 0.1*s.opts.ϵ_tol
                        @logmsg InnerLoop "<interior-point solve complete>: small search direction"
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
                    if s.θ < s.opts.ϵ_tol
                        # @logmsg InnerLoop "Infeasibility detected"
                        @warn "Infeasibiity detected"
                        return
                    else
                        augment_filter!(s)
                        restoration!(solver.s̄,s)
                    end
                else  # successful line search
                    augment_filter!(s)
                    accept_step!(s)
                end
            end

            s.opts.z_reset && reset_z!(s)

            # Calculate everything at the new trial point
            eval_step!(s)

            if s.opts.verbose
                println("j: $(s.j), k: $(s.k)")
                s.model.n < 5 ? println("   x: $(s.x)") : nothing
                println("")
                println("θ: $(s.θ), φ: $(s.φ)")
                println("Eμ: $(eval_Eμ(s.μ,s))")
            end


            s.k += 1
            if s.k > s.opts.max_iter
                # TODO: don't throw an error! handle this gracefully
                @warn "max iterations"
                return
            end
        end  # inner while loop


        if eval_Eμ(0.0,s) <= s.opts.ϵ_tol && norm(s.c_al,1) <= s.opts.ϵ_al_tol
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
        println("<augmented-Lagrangian interior-point solve>:complete\n")
        s.model.n < 5 ? println("   x: $(s.x)") : nothing
        println("   f: $(s.f)")
        println("   θ: $(s.θ), φ: $(s.φ)")
        println("   E0: $(eval_Eμ(0.0,s))")
        println("   norm(c): $(norm(s.c[s.c_al_idx .== 0]))")
        println("   norm(c_al): $(norm(s.c_al))")
    end
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
    @logmsg InnerLoop :f value=s.f width=10
    @logmsg InnerLoop :μ value=s.μ width=10
    @logmsg InnerLoop :α value=s.α
end
