function solve!(s::Solver; verbose=false)
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
            search_direction!(s)

            if !line_search(s)
                augment_filter!(s)
                restoration!(s)
            else
                augment_filter!(s)
                update!(s)
            end

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

        check_bnds(s)

        if s.k == 0
            update_μ!(s)
            update_τ!(s)
            eval_barrier!(s)
            s.j += 1
            empty!(s.filter)
            push!(s.filter,(s.θ_max,Inf))

            check_bnds(s)
        end
    end
    verbose ? println("<interior-point solve complete>") : nothing
end
