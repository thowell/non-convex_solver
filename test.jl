include("src/interior_point.jl")

n = 5000
m = 1000

x0 = ones(n)

xL = -Inf*ones(n)
xL[1] = -10.
xL[2] = -5.
xU = Inf*ones(n)
xU[5] = 20.

f_func(x) = x'*x
function ∇f_func!(∇f,x)
    ∇f .= 2*x
    return nothing
end
function ∇²f_func!(∇²f,x)
    view(∇²f,CartesianIndex.(1:n,1:n)) .= 2.0
    return nothing
end

f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = x[1:m].^2 .- 1.2
function c_func!(c,x)
    c .= x[1:m].^2 .- 1.2
    return nothing
end
function ∇c_func!(∇c,x)
    ∇c[:,1:m] = 2.0*Diagonal(x[1:m])
    return nothing
end
function ∇²cy_func!(∇²c,x,y)
    ∇²c[1:m,1:m] = 2.0*Diagonal(y)
    return nothing
end
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f_func,∇f_func!,∇²f_func!,c_func!,∇c_func!,∇²cy_func!)

opts = Options{Float64}(kkt_solve=:symmetric,
                        iterative_refinement=true,
                        verbose=false)

s = InteriorPointSolver(x0,model,opts=opts)
@time solve!(s)
