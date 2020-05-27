using Test
using BenchmarkTools
using NLPModels
using CUTEst
# using Ipopt

include("../src/interior_point.jl")
include("test_utils.jl")

include("cutest_tests.jl")
include("complementarity_tests.jl")
include("contact_tests.jl")
include("nlp_tests.jl")
