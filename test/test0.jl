n = 50
m = 30 + 3

x0 = ones(n)

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = x'*x
c_func(x) = [x[1:30].^2 .- 1.2; 
    x[1] + 10.0; x[2] + 5.0; 20.0 - x[5]]

model = Model(n,m,xL,f_func,c_func,cI_idx=[zeros(Bool,30); ones(Bool,3)], cA_idx=zeros(Bool,m))#,cA_idx=[ones(Bool,30); zeros(Bool,3)])

options = Options{Float64}(
                        linear_solve_type=:symmetric,
                        iterative_refinement=false,
                        residual_tolerance=1.0e-5,
                        equality_tolerance=1.0e-5,
                        max_residual_iterations=250,
                        verbose=true,
                        linear_solver=:QDLDL,
                        )

s = Solver(x0,model,options=options)
@time solve!(s)
