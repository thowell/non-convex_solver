include("problems.jl")
opts = Options{Float64}(
						kkt_solve=:symmetric,
						linear_solver=:QDLDL,
                        max_iter=1000,
                        verbose=true,
						ϵ_tol=1.0e-5,
						ϵ_al_tol=1.0e-5,
						# quasi_newton=:bfgs
						)
@testset "NLP problems" begin
	solver = NCSolver(wachter()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NCSolver(wachter_reform()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NCSolver(problem1()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NCSolver(problem2()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NCSolver(problem3()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NCSolver(problem4()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NCSolver(problem6()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NCSolver(problem7()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = NCSolver(problem8()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol
end
