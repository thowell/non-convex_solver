
status = Symbol[]
@testset "CUTEst inter Problems" begin
    for prob in CUTEstProbs[:small]
        println(prob)
        nlp = CUTEstModel(prob)
        model = Model(nlp)
        opts = Options{Float64}(kkt_solve=:symmetric,
                                ϵ_tol = 1e-8,
                                ϵ_al_tol = 1e-8,
                                verbose=false)
        solver = InteriorPointSolver(nlp.meta.x0, model,opts=opts)
        try
            solve!(solver)
            s = solver.s
            @test eval_Eμ(0.0,s) <= s.opts.ϵ_tol
            # @test norm(view(s.x,get_r_idx(s)),1) <= s.opts.ϵ_al_tol
            push!(status, :solved)
        catch
            push!(status, :failed)
        end
        finalize(nlp)
    end

    @test count(status .!= :failed)/length(status) > .9
end
