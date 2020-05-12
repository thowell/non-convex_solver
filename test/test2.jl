include("../src/interior_point.jl")


n = 2
m = 0
x0 = [1.; 0.1]

xL = -Inf*ones(n)
xU = [0.25; 0.25]

f_func(x) = (x[1] + 0.5)^2 + (x[2] - 0.5)^2
f, ∇f!, ∇²f! = objective_functions(f_func)

function c_func!(c,x,model::AbstractModel)
    return nothing
end
function ∇c_func!(∇c,x,model::AbstractModel)
    return nothing
end

function ∇²cy_func!(∇²c,x,y,model::AbstractModel)
    return nothing
end

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c_func!,∇c_func!,∇²cy_func!)
s = InteriorPointSolver(x0,model)
@time solve!(s)
