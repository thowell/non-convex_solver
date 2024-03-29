module NonConvexSolver

using LinearAlgebra, ForwardDiff, SparseArrays, Parameters
using SolverLogging, Logging, Crayons
# using HSL
using QDLDL
# using Pardiso

include("options.jl")
include("indices.jl")
include("views.jl")
include("bounds.jl")
include("model.jl")
include("scaling.jl")
include("quasi_newton.jl")
include("linear_solver.jl")
include("solver.jl")
include("filter.jl")
include("second_order_correction.jl")
include("kkt_error_reduction.jl")
include("restoration.jl")
include("inertia_correction.jl")
include("iterative_refinement.jl")
include("search_direction.jl")
include("watch_dog.jl")
include("line_search.jl")
include("augmented_lagrangian.jl")
include("solve.jl")
include("utils.jl")

end # module
