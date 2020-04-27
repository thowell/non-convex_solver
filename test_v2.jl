include("src/interior_point.jl")

n = 500
m = 100

x0 = ones(n)

xL = -Inf*ones(n)
xL[1] = -10.
xL[2] = -5.
xU = Inf*ones(n)
xU[5] = 20.

f_func(x,model::AbstractModel) = x'*x
function ∇f_func!(∇f,x,model::AbstractModel)
    ∇f .= 2*x
    return nothing
end
function ∇²f_func!(∇²f,x,model::AbstractModel)
    view(∇²f,CartesianIndex.(1:n,1:n)) .= 2.0
    return nothing
end

function c_func!(c,x,model::AbstractModel)
    c .= x[1:m].^2 .- 1.2
    return nothing
end
function ∇c_func!(∇c,x,model::AbstractModel)
    ∇c[:,1:m] = 2.0*Diagonal(x[1:m])
    return nothing
end
function ∇²cy_func!(∇²c,x,y,model::AbstractModel)
    ∇²c[1:m,1:m] = 2.0*Diagonal(y)
    return nothing
end

model = Model(n,m,xL,xU,f_func,∇f_func!,∇²f_func!,c_func!,∇c_func!,∇²cy_func!)

opts = Options{Float64}(kkt_solve=:symmetric,
                        iterative_refinement=true,
                        verbose=false)

s = InteriorPointSolver(x0,model,opts=opts)
@time solve!(s)
