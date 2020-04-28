function solve!(solver::InteriorPointSolver)
    # TODO: some sort of reset?
    s = solver.s

    # evaluate problem
    eval_iterate!(s)

    # initialize filter
    push!(s.filter,(s.θ_max,Inf))

    if s.opts.verbose
        println("<interior-point solve>\n")
        println("θ0: $(s.θ), φ0: $(s.φ)")
        println("Eμ0: $(eval_Eμ(0.0,s))")
    end

    while eval_Eμ(0.0,s) > s.opts.ϵ_tol

        # Converge the interior point sub-problem
        while eval_Eμ(s.μ,s) > s.opts.κϵ*s.μ
            s.opts.relax_bnds ? relax_bnds!(s) : nothing

            # solve for the search direction and check if it's small
            if search_direction!(s)
                s.small_search_direction_cnt += 1
                if s.small_search_direction_cnt == s.opts.small_search_direction_max
                    if s.μ < 0.1*s.opts.ϵ_tol
                        s.opts.verbose ? println("<interior-point solve complete>: small search direction") : nothing
                        return
                    else
                        break
                    end
                end
                α_max!(s)
                αz_max!(s)
                augment_filter!(s)
                update!(s)  # TODO: maybe call this something a little more informative?
            else
                s.small_search_direction_cnt = 0

                # Perform line search and check if it fails
                if !line_search(s)
                    if s.θ < s.opts.ϵ_tol
                        @warn "infeasibility detected"
                        return
                    else
                        augment_filter!(s)
                        # @warn "updating y_al"
                        # s.λ .= s.λ + s.ρ*s.c_al

                        restoration!(solver.s̄,s)
                    end
                else  # successful line search
                    augment_filter!(s)
                    update!(s)
                end
            end

            s.c_tmp .= copy(s.c)
            s.opts.z_reset ? reset_z!(s) : nothing

            # Calculate everything at the new trial point
            eval_iterate!(s)

            s.k += 1
            if s.k > s.opts.max_iter
                error("max iterations")
            end

            if s.opts.verbose
                println("iteration ($(s.j),$(s.k)):")
                s.model.n < 5 ? println("x: $(s.x)") : nothing
                println("θ: $(θ(s.x,s)), φ: $(barrier(s.x,s))")
                println("Eμ: $(eval_Eμ(s.μ,s))")
                println("E0: $(eval_Eμ(0.0,s))")

                println("α: $(s.α)\n")
            end
        end  # while

        if eval_Eμ(0.0,s) <= s.opts.ϵ_tol && norm(s.c_al,1) <= s.opts.ϵ_al_tol
            break
        else

            update_μ!(s)
            update_τ!(s)

            s.λ .= s.λ + s.ρ*s.c_al
            s.ρ = 1.0/s.μ

            eval_iterate!(s)

            # eval_barrier!(s)
            s.j += 1
            empty!(s.filter)
            push!(s.filter,(s.θ_max,Inf))

            if s.k == 0
                update_μ!(s)
                update_τ!(s)

                s.λ .= s.λ + s.ρ*s.c_al
                s.ρ = 1.0/s.μ

                # eval_barrier!(s)
                eval_iterate!(s)

                s.j += 1
                empty!(s.filter)
                push!(s.filter,(s.θ_max,Inf))
            end
        end
    end
    if s.opts.verbose
        println("<interior-point solve complete>")
        println("   iteration ($(s.j),$(s.k)):")

        s.model.n < 5 ? println("x: $(s.x)") : nothing
        println("   θ: $(θ(s.x,s)), φ: $(barrier(s.x,s))")
        println("   E0: $(eval_Eμ(0.0,s))")
        println("   f: $(s.f)")
        println("   norm(c): $(norm(s.c[s.c_al_idx .== 0]))")
        println("   norm(c_al): $(norm(s.c_al))")
    end
end
