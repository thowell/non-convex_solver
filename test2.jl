include("src/interior_point.jl")

n = 2
m = 0
x0 = [1.; 0.1]

xL = -Inf*ones(n)
xU = [0.25; 0.25]#Inf*ones(n)

f_func(x) = (x[1] + 0.5)^2 + (x[2] - 0.5)^2
function ∇f_func!(∇f,x)
    ∇f .= ForwardDiff.gradient(f_func,x)
end
function ∇²f_func!(∇²f,x)
    ∇²f .= ForwardDiff.hessian(f_func,x)
end

c_func(x) = zeros(m)
∇c_func(x) = zeros(m,n)
∇²cλ_func(x,λ) = zeros(n,n)

s = InteriorPointSolver(x0,n,m,xL,xU,f_func,∇f_func!,∇²f_func!,c_func,∇c_func,∇²cλ_func,opts=Options{Float64}(max_iter=500))
@time solve!(s,verbose=true)
