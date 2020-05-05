include("../src/interior_point.jl")


n = 12
m = 7

x0 = ones(n)

xL = zeros(n)
xU = Inf*ones(n)


f_func(x) = (x[1] - 5)^2 + (2*x[2] + 1)^2 + 10.0*x[12]
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = [2*(x[2] - 1) - 1.5*x[2] + x[3] - 0.5*x[4] + x[5];
             3*x[1] - x[2] - 3.0 - x[6];
             -x[1] + 0.5*x[2] + 4.0 - x[7];
             -x[1] - x[2] + 7.0 - x[8];
             x[12] - x[3]*x[6] - x[9];
             x[12] - x[4]*x[7] - x[10];
             x[12] - x[5]*x[8] - x[11]]
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)

s = InteriorPointSolver(x0,model,opts=Options{Float64}(kkt_solve=:symmetric,relax_bnds=false,single_bnds_damping=true,iterative_refinement=true,max_iter=100))
@time solve!(s)
x = s.s.x

x[3]
x[6]

x[4]
x[7]

x[5]
x[8]