module NonConvexSolver

using LinearAlgebra
using SparseArrays
using Crayons
using QDLDL
using Symbolics
using ForwardDiff


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

include(joinpath("../src_alt/", "generate.jl"))
include(joinpath("../src_alt/", "indices.jl"))
include(joinpath("../src_alt/", "data.jl"))
include(joinpath("../src_alt/", "dimensions.jl"))
include(joinpath("../src_alt/", "problem.jl"))
include(joinpath("../src_alt/","solver.jl"))
include(joinpath("../src_alt/","initialize.jl"))

end # module
