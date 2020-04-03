include("src/interior_point.jl")

n = 2
m = 0
x0 = [1.; 0.1]

xL = -Inf*ones(n)
xU = [0.25; 0.25]#Inf*ones(n)

f_func(x) = (x[1] + 0.5)^2 + (x[2] - 0.5)^2
c_func(x) = zeros(m)

∇f_func(x) = ForwardDiff.gradient(f_func,x)
∇c_func(x) = zeros(m,n)

function ∇²L_func(x,λ)
    ∇L(x) = ∇f_func(x) + ∇c_func(x)'*λ
    return ForwardDiff.jacobian(∇L,x)
end

s = InteriorPointSolver(x0,n,m,xL,xU,f_func,c_func,∇f_func,∇c_func,∇²L_func; opts=Options{Float64}(max_iter=1e3))
@time solve!(s,verbose=true)
