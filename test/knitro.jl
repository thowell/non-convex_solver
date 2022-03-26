n = 8
m = 7 + n

x0 = zeros(n)

# xL = zeros(n)
xL = -Inf * ones(n)
xU = Inf*ones(n)

f_func(x) = (x[1] - 5)^2 + (2*x[2] + 1)^2
c_func(x) = [2*(x[2] - 1) - 1.5*x[2] + x[3] - 0.5*x[4] + x[5];
                3*x[1] - x[2] - 3.0 - x[6];
                -x[1] + 0.5*x[2] + 4.0 - x[7];
                -x[1] - x[2] + 7.0 - x[8];
                x[3]*x[6];
                x[4]*x[7];
                x[5]*x[8];
                x]

f, ∇f!, ∇²f! = objective_functions(f_func)
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

cI_idx = [zeros(Bool,7); ones(Bool, n)]
cA_idx = [ones(Bool,7); zeros(Bool,n)]
# cA_idx[5:end] .= 1
# cA_idx = ones(Bool,m)

model = Model(n,m,xL,xU,
                f,∇f!,∇²f!,
                c!,∇c!,∇²cy!,
                cI_idx=cI_idx,cA_idx=cA_idx)

options = Options{Float64}(
                        linear_solve_type=:symmetric,
                        max_residual_iterations=1000,
                        residual_tolerance=1.0e-5,
                        equality_tolerance=1.0e-5,
                        verbose=true,
                        linear_solver=:QDLDL,
                        )

s = Solver(x0,model,options=options)
@time solve!(s)

x = s.x
x[3]
x[6]

x[4]
x[7]

x[5]
x[8]
