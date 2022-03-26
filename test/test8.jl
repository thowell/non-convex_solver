n = 3
m = 1

x0 = rand(n)

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = x[1] - 2*x[2] + x[3] + sqrt(6)
f, ∇f!, ∇²f! = objective_functions(f_func)

c_func(x) = [1 - x[1]^2 - x[2]^2 - x[3]^2]
c!, ∇c!, ∇²cy! = constraint_functions(c_func)

model = Model(n,m,xL,xU,f_func,c_func,cI_idx=ones(Bool,m),cA_idx=zeros(Bool,m))

s = Solver(x0,model,options=Options{Float64}(linear_solve_type=:symmetric,
                                                        residual_tolerance=1.0e-5,
                                                        equality_tolerance=1.0e-5,
                                                        linear_solver=:QDLDL,
                                                        verbose=true,
                                                        max_residual_iterations=250
                                                        ))
@time solve!(s)
