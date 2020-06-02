include("contact_problems.jl")

@testset "Contact problems" begin
    # particle
	@info "particle"
	solver = InteriorPointSolver(particle()...,opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

	solver = InteriorPointSolver(
              particle(qpp=[0.,0.,10.],v0=[10.,-7.0, 0.])...,
              opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

    solver = InteriorPointSolver(
              particle(qpp=[0.,0.,0.01],v0=[1.,0., 0.])...,
              opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

    solver = InteriorPointSolver(
              particle(qpp=[0.,0.,0.01],v0=[1.,1., 0.])...,
              opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

    solver = InteriorPointSolver(
              particle(qpp=[0.,0.,0.01],v0=[0.,0., 0.])...,
              opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

    solver = InteriorPointSolver(
              particle(qpp=[0.,0.,-1.0],v0=[0.,0., 0.])...,
              opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

    # Raibert hopper
	@info "raibert hopper"
    solver = InteriorPointSolver(
              hopper()...,
              opts=opts)
	solve!(solver)
	@test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

    solver = InteriorPointSolver(
              hopper(T=5)...,
              opts=opts)
    solve!(solver)
    @test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol

    solver = InteriorPointSolver(
              hopper(T=5,r=0.7,q0=[0.,0.7,0.7,0.,0.],qf =[0.,1.0+0.7,0.7,0.,0.])...,
              opts=opts)
    solve!(solver)
    @test eval_Eμ(0.0,solver.s) <= solver.s.opts.ϵ_tol
end
