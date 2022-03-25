n = 2
m = 2

x0 = rand(n)

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = 100*(x[2]-x[1]^2)^2 + (1-x[1])^2

c_func(x) = [-(x[1] -1)^3 + x[2] - 1;
             -x[1] - x[2] + 2]

model = Model(n,m,xL,xU,f_func,c_func,cI_idx=ones(Bool,m),cA_idx=zeros(Bool,m))

s = Solver(x0,model,opts=opts=Options{Float64}(kkt_solve=:symmetric,
                                                        residual_tolerance=1.0e-5,
                                                        equality_tolerance=1.0e-5,
                                                        verbose=true,
                                                        linear_solver=:QDLDL,
                                                        max_residual_iterations=500,
                                                        ))
@time solve!(s)
