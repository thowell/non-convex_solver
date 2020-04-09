include("src/interior_point.jl")

n = 2
m = 0
x0 = [1.; 0.1]

xL = -Inf*ones(n)
xU = [0.25; 0.25]

f_func(x) = (x[1] + 0.5)^2 + (x[2] - 0.5)^2
f, ∇f!, ∇²f! = objective_functions(f_func)

function c_func!(c,x)
    return nothing
end
function ∇c_func!(∇c,x)
    return nothing
end

function ∇²cλ_func!(∇²c,x,λ)
    return nothing
end

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c_func!,∇c_func!,∇²cλ_func!)

s = InteriorPointSolver(x0,model,opts=Options{Float64}(iterative_refinement=false,kkt_solve=:unreduced,max_iter=500))
@time solve!(s,verbose=true)
