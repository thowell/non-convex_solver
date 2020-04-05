include("src/interior_point.jl")

n = 3
m = 2

x0 = [-2.0;3.0;1.0]

xL = -Inf*ones(n)
xL[2] = 0.
xL[3] = 0.
xU = Inf*ones(n)

f_func(x) = x[1]
function ∇f_func!(∇f,x)
    ∇f .= ForwardDiff.gradient(f_func,x)
end
function ∇²f_func!(∇²f,x)
    ∇²f .= ForwardDiff.hessian(f_func,x)
end

c_func(x) = [x[1]^2 - x[2] - 1.0;
             x[1] - x[3] - 0.5]

function c_func(c,x)
    c .= c_func(x)
    return nothing
end

function ∇c_func(∇c,x)
    ∇c .= ForwardDiff.jacobian(c_func,x)
    return nothing
end

function ∇²cλ_func(∇²cλ,x,λ)
    ∇c_func(x) = ForwardDiff.jacobian(c_func,x)
    ∇cλ(x) = ∇c_func(x)'*λ
    ∇²cλ .= ForwardDiff.jacobian(∇cλ,x)
    return return nothing
end

s = InteriorPointSolver(x0,n,m,xL,xU,f_func,∇f_func!,∇²f_func!,c_func,∇c_func,∇²cλ_func,opts=Options{Float64}(max_iter=500))
@time solve!(s,verbose=true)
