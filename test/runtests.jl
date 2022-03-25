using Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))
using NonConvexSolver 

include("nlp/test0.jl")
include("nlp/test1.jl")
include("nlp/test6.jl")
include("nlp/test7.jl")
include("nlp/test8.jl")
include("nlp/test9.jl")
include("complementarity/test_knitro_comp.jl")