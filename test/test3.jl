include("../src/interior_point.jl")

n = 2
m = 1
x0 = [-2.; 10.]

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = 2.0*(x[1]^2 + x[2]^2 - 1.0) - x[1]
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = [x[1]^2 + x[2]^2 - 1.0]
c_func_d(x) = x[1]^2 + x[2]^2 - 1.0
function c_func!(c,x,model)
    c .= c_func(x)
    return nothing
end
∇c_func(x) = Array(ForwardDiff.gradient(c_func_d,x)')

function ∇²cy_func(x,y)
    ∇cy(x) = ∇c_func(x)'*y
    return ForwardDiff.jacobian(∇cy,x)
end

function ∇c_func!(∇c,x,model)
    ∇c .= ∇c_func(x)
    return nothing
end

function ∇²cy_func!(∇²cy,x,y,model)
    ∇²cy .= ∇²cy_func(x,y)
    return nothing
end

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c_func!,∇c_func!,∇²cy_func!)

s = InteriorPointSolver(x0,model,opts=Options{Float64}(iterative_refinement=true,
                        kkt_solve=:symmetric,
                        nlp_scaling=true,
                        relax_bnds=true))
@time solve!(s)
norm(c_func(s.s.x),1)
