include("nlp_problems.jl")
opts = Options{Float64}(kkt_solve=:symmetric,
                        max_iter=1000,
                        verbose=false)
@testset "NLP problems" begin
	solver = NonConvexSolver(wachter()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NonConvexSolver(wachter_reform()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NonConvexSolver(problem1()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NonConvexSolver(problem2()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NonConvexSolver(problem3()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NonConvexSolver(problem4()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NonConvexSolver(problem6()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NonConvexSolver(problem7()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NonConvexSolver(problem8()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol
end
