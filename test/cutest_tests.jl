status = Symbol[]
@testset "CALIPSO - CUTEst small problems" begin
    for prob in CUTEstProbs[:small]
        println(prob)
        nlp = CUTEstModel(prob)
        model = Model(nlp)
        opts = Options{Float64}(kkt_solve=:symmetric,
                                max_iter=1000,
                                ϵ_tol = 1e-8,
                                ϵ_al_tol = 1e-8,
                                verbose=true)
        solver = InteriorPointSolver(nlp.meta.x0, model,opts=opts)
        # solver.s.ρ = 1.0
        try
            solve!(solver)
            s = solver.s
            @test eval_Eμ(0.0,s) <= s.opts.ϵ_tol
            push!(status, :solved)
        catch
            @test eval_Eμ(0.0,s) <= s.opts.ϵ_tol

            push!(status, :failed)
        end
        finalize(nlp)
    end

    @test count(status .!= :failed)/length(status) > .9
end

@testset "Ipopt - CUTEst small problems" begin
    for prob in CUTEstProbs[:small]
        println(prob)
        nlp = CUTEstModel(prob)
        ipoptProb = createProblem(nlp)
        ipoptProb.x .= copy(nlp.meta.x0)

        st = solveProblem(ipoptProb)
        @test st == 0
        finalize(nlp)
    end

    @test count(status .!= :failed)/length(status) > .9
end
