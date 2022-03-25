status = Symbol[]
include("cutest_problems_small.jl")
include("cutest_problems_medium.jl")
@testset "CUTEst (small)" begin
    for prob in small
        println("problem: $(prob)")
        nlp = SlackModel(CUTEstModel(prob))
        model = Model(nlp)
        solver = NCSolver(nlp.meta.x0, model,opts=opts)
        try
            solve!(solver)
            @test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol
        catch
            @test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol
        end
        finalize(nlp)
    end
end
#
# @testset "CUTEst (medium)" begin
#     for prob in medium
        # println("problem: $(prob)")
#         nlp = SlackModel(CUTEstModel(prob))
#         model = Model(nlp)
#         solver = NCSolver(nlp.meta.x0, model,opts=opts)
#         try
#             solve!(solver)
#             @test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol
#         catch
#             @test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol
#         end
#         finalize(nlp)
#     end
# end

# @testset "Ipopt - CUTEst small problems" begin
#     for prob in small
#         println("problem: $(prob)")
#         nlp = CUTEstModel(prob)
#         ipoptProb = createProblem(nlp)
#         ipoptProb.x .= copy(nlp.meta.x0)
#         # addOption(ipoptProb,"hessian_approximation","limited-memory")
#         st = solveProblem(ipoptProb)
#         @test st == 0
#         finalize(nlp)
#     end
# end

# @testset "Ipopt - CUTEst medium problems" begin
#     for prob in medium
#         println("problem: $(prob)")
#         nlp = CUTEstModel(prob)
#         ipoptProb = createProblem(nlp)
#         ipoptProb.x .= copy(nlp.meta.x0)
#
#         st = solveProblem(ipoptProb)
#         @test st == 0
#         finalize(nlp)
#     end
#
#     @test count(status .!= :failed)/length(status) > .9
# end
