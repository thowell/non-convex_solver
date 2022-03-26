using Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))
using NonConvexSolver 

include("test0.jl")
include("wachter.jl")
include("test6.jl")
include("test7.jl")
include("test8.jl")
include("maratos.jl")
include("knitro.jl")

