using Test
using BenchmarkTools
using Ipopt
using NLPModels

include("../src/interior_point.jl")
include("test_utils.jl")
include("cutest.jl")

include("cutest_tests.jl")
