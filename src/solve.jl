function solve!(solver::InteriorPointSolver; verbose=false)
    s = solver.s

    # evaluate problem
    eval_iterate!(s)

    # initialize filter
    push!(s.filter,(s.θ_max,Inf))

    if verbose
        println("<interior-point solve>\n")
        println("φ0: $(s.φ), θ0: $(s.θ)")
        println("Eμ0: $(eval_Eμ(0.0,s))")
    end

    while eval_Eμ(0.0,s) > s.opts.ϵ_tol
        while eval_Eμ(s.μ,s) > s.opts.κϵ*s.μ
            s.opts.relax_bnds ? relax_bnds!(s) : nothing
            if search_direction!(s)
                s.small_search_direction_cnt += 1
                if s.small_search_direction_cnt == s.opts.small_search_direction_max
                    if s.μ < 0.1*s.opts.ϵ_tol
                        verbose ? println("<interior-point solve complete>: small search direction") : nothing
                        return
                    else
                        break
                    end
                end
                α_max!(s)
                αz_max!(s)
                augment_filter!(s)
                update!(s)
            else
                s.small_search_direction_cnt = 0

                if !line_search(s)
                    augment_filter!(s)
                    restoration!(solver.s̄,s)
                    augment_filter!(s)
                else
                    augment_filter!(s)
                    update!(s)
                end
            end

            reset_z!(s)
            eval_iterate!(s)

            s.k += 1
            if s.k > s.opts.max_iter
                error("max iterations")
            end

            if verbose
                println("iteration (j,k): ($(s.j),$(s.k))")
                println("x: $(s.x)")
                println("θjk: $(θ(s.x,s)), φjk: $(barrier(s.x,s))")
                println("Eμ: $(eval_Eμ(s.μ,s))")
                println("α: $(s.α)\n")
            end
        end

        update_μ!(s)
        update_τ!(s)
        eval_barrier!(s)
        s.j += 1
        empty!(s.filter)
        push!(s.filter,(s.θ_max,Inf))

        if s.k == 0
            update_μ!(s)
            update_τ!(s)
            eval_barrier!(s)
            s.j += 1
            empty!(s.filter)
            push!(s.filter,(s.θ_max,Inf))
        end
    end
    verbose ? println("<interior-point solve complete>") : nothing
end
