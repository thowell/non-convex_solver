using BenchmarkTools

function benchmark_solve!(solver::NonConvexSolver; samples=10, evals=10)
    x0 = copy(solver.s.x)
    b = @benchmark begin
        reset_solver!($solver.s, $x0)
        solve!($solver)
    end samples=samples evals=evals
    return b
end
