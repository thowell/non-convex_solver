module NonConvexSolver

using LinearAlgebra, ForwardDiff, SparseArrays, Parameters
using Crayons
using QDLDL

include("options.jl")
include("indices.jl")
include("views.jl")
include("bounds.jl")
include("model.jl")
include("scaling.jl")
include("linear_solver.jl")
include("solver.jl")
include("filter.jl")
include("second_order_correction.jl")
include("inertia_correction.jl")
include("iterative_refinement.jl")
include("search_direction.jl")
include("line_search.jl")
include("augmented_lagrangian.jl")
include("solve.jl")

end # module
