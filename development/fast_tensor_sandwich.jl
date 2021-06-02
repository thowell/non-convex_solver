using FiniteDiff, SparseArrays, StaticArrays, BenchmarkTools

fcalls = 0
function f(dx,x) # in-place
  global fcalls += 1
  for i in 2:length(x)-1
    dx[i] = x[i-1] - 2x[i] + x[i+1]
  end
  dx[1] = -2x[1] + x[2]
  dx[end] = x[end-1] - 2x[end]
  nothing
end

handleleft(x,i) = i==1 ? zero(eltype(x)) : x[i-1]
handleright(x,i) = i==length(x) ? zero(eltype(x)) : x[i+1]
function g(x) # out-of-place
  global fcalls += 1
  len_x = length(x)
  SVector{len_x}([handleleft(x,i) - 2*x[i] + handleright(x,i) for i in 1:len_x])
end

x = @SVector rand(10)
FiniteDiff.finite_difference_jacobian(g,x)

x = rand(1000)
# output = zeros(1000,1000)
output2 = spzeros(1000,1000)
output3 = view(output2,1:1000,1:1000)
@benchmark FiniteDiff.finite_difference_jacobian!($output3,$f,$x)
cache = FiniteDiff.JacobianCache(x)
@benchmark FiniteDiff.finite_difference_jacobian!($output3,$f,$x,$cache)
