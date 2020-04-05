include("src/interior_point.jl")

n = 20
m = 10
x0 = ones(n)
xL = -Inf*ones(n)
xL[1] = -10.
xL[2] = -5.
xU = Inf*ones(n)
xU[5] = 20.
f_func(x) = x'*x
∇f_func(x) = ForwardDiff.gradient(f_func,x)
∇²f_func(x) = ForwardDiff.hessian(f_func,x)

c_func(x) = x[1:m].^2 .- 1.2
∇c_func(x) = ForwardDiff.jacobian(c_func,x)


function ∇²cλ_func(x,λ)
    ∇cλ(x) = ∇c_func(x)'*λ
    return ForwardDiff.jacobian(∇cλ,x)
end

s = InteriorPointSolver(x0,n,m,xL,xU,f_func,∇f_func,∇²f_func,c_func,∇c_func,∇²cλ_func,opts=Options{Float64}(max_iter=500))
@time solve!(s,verbose=true)
