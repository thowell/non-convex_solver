
status = Symbol[]
@testset "CUTEst Small Problems" begin
    for prob in CUTEstProbs[:small] 
        println(prob)
        nlp = CUTEstModel(prob)
        tol = 1e-8
        model = Model(nlp)
        opts = Options{Float64}(kkt_solve=:symmetric,
                                iterative_refinement=true,
                                relax_bnds=true,
                                max_iter=100,
                                y_init_ls=true,
                                max_iterative_refinement=1000,
                                ϵ_tol = 1e-8,
                                verbose=false)
        solver = InteriorPointSolver(nlp.meta.x0, model, c_al_idx=zeros(Bool,nlp.meta.ncon),opts=opts)
        try
            solve!(solver)
            eval_step!(solver.s)
            @test norm(solver.s.∇L, Inf) < tol
            @test norm(solver.s.c, Inf) < tol
            @test norm(min.(solver.s.x - solver.s.xL, 0), Inf) < tol
            @test norm(max.(solver.s.x - solver.s.xU, 0), Inf) < tol
            push!(status, :solved)
        catch
            push!(status, :failed)
        end
        finalize(nlp)
    end

    @test count(status .!= :failed)/length(status) > .9
end
