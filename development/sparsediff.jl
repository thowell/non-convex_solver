using ForwardDiff, FiniteDiff, SparsityDetection, SparseArrays, SparseDiffTools
using BenchmarkTools

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

function g(x) # out-of-place
  global fcalls += 1
  dx = zero(x)
  for i in 2:length(x)-1
    dx[i] = x[i-1] - 2x[i] + x[i+1]
  end
  dx[1] = -2x[1] + x[2]
  dx[end] = x[end-1] - 2x[end]
  dx
end

x = zeros(100)
dx = zero(x)

f(dx,x)
fcalls
g(x)
FiniteDiff.finite_difference_jacobian(g,x)
fcalls
@benchmark FiniteDiff.finite_difference_jacobian($g,$x)

x = rand(100)
y = similar(x)

sparsity_pattern = jacobian_sparsity(f,y,x)
jac = Float64.(sparse(sparsity_pattern))
colors = matrix_colors(jac)
maximum(colors)
@benchmark FiniteDiff.finite_difference_jacobian!($jac,$f,$x,colorvec=$colors)
@benchmark ForwardDiff.jacobian($f,$dx,$x)
j_cache = ForwardColorJacCache(f,x,colorvec=colors,sparsity=sparsity_pattern)
@benchmark forwarddiff_color_jacobian!($jac,$f,$x,$j_cache)
