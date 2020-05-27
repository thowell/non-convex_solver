using Test
using BenchmarkTools
using Ipopt
using NLPModels
using CUTEst

include("../src/interior_point.jl")
include("test_utils.jl")
include("cutest_tests.jl")
