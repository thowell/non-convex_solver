using Test
using BenchmarkTools
using NLPModels
using CUTEst
using Ipopt

include("../src/non-convex_solver.jl")
include("test_utils.jl")

opts = Options{Float64}(kkt_solve=:symmetric,
                        max_iter=1000,
                        verbose=false,
                        iterative_refinement=true,
                        quasi_newton=:none)

include("cutest_tests.jl")
include("complementarity_tests.jl")
include("contact_tests.jl")
include("nlp_tests.jl")

# nlp = CUTEstModel("HS77")
# ipoptProb = createProblem(nlp)
# ipoptProb.x .= copy(nlp.meta.x0)
# addOption(ipoptProb,"hessian_approximation","limited-memory")
# st = solveProblem(ipoptProb)
# finalize(nlp)
