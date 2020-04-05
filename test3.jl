include("src/interior_point.jl")

n = 2
m = 1
x0 = [-2.; 10.]

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = 2.0*(x[1]^2 + x[2]^2 - 1.0) - x[1]
∇f_func(x) = ForwardDiff.gradient(f_func,x)
∇²f_func(x) = ForwardDiff.hessian(f_func,x)

c_func(x) = [x[1]^2 + x[2]^2 - 1.0]
c_func_d(x) = x[1]^2 + x[2]^2 - 1.0
∇c_func(x) = Array(ForwardDiff.gradient(c_func_d,x)')

function ∇²cλ_func(x,λ)
    ∇cλ(x) = ∇c_func(x)'*λ
    return ForwardDiff.jacobian(∇cλ,x)
end

s = InteriorPointSolver(x0,n,m,xL,xU,f_func,∇f_func,∇²f_func,c_func,∇c_func,∇²cλ_func; opts=Options{Float64}())
@time solve!(s,verbose=true)
