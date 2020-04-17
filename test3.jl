include("src/interior_point.jl")

n = 2
m = 1
x0 = [-2.; 10.]

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = 2.0*(x[1]^2 + x[2]^2 - 1.0) - x[1]
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = [x[1]^2 + x[2]^2 - 1.0]
c_func_d(x) = x[1]^2 + x[2]^2 - 1.0
function c_func!(c,x)
    c .= c_func(x)
    return nothing
end
∇c_func(x) = Array(ForwardDiff.gradient(c_func_d,x)')

function ∇²cλ_func(x,λ)
    ∇cλ(x) = ∇c_func(x)'*λ
    return ForwardDiff.jacobian(∇cλ,x)
end

function ∇c_func!(∇c,x)
    ∇c .= ∇c_func(x)
    return nothing
end

function ∇²cλ_func!(∇²cλ,x,λ)
    ∇²cλ .= ∇²cλ_func(x,λ)
    return nothing
end

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c_func!,∇c_func!,∇²cλ_func!)

s = InteriorPointSolver(x0,model,opts=Options{Float64}(iterative_refinement=true,
                        kkt_solve=:symmetric,
                        nlp_scaling=true,
                        relax_bnds=true))
# s.s.ρ = 10.0
@time solve!(s,verbose=true)
norm(c_func(s.s.x),1)

# s_new = InteriorPointSolver(s.s.x,model,opts=opts)
# s_new.s.λ .= s.s.λ
# s_new.s.λ_al .= s.s.λ_al + s.s.ρ*s.s.c
# s_new.s.ρ = s.s.ρ*10.0
# solve!(s_new,verbose=true)
# s = s_new
# norm(c_func(s.s.x),1)
