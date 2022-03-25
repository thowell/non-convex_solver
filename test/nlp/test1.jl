# NOTE: Ipopt fails to converge on this problem.
n = 3
m = 2

x0 = [-2.0;3.0;1.0]

xL = -Inf*ones(n)
xL[2] = 0.
xL[3] = 0.
xU = Inf*ones(n)

f_func(x) = x[1]

c_func(x) = [x[1]^2 - x[2] - 1.0;
             x[1] - x[3] - 0.5]

model = Model(n,m,xL,xU,f_func,c_func,cA_idx=ones(Bool,m))

opts = Options{Float64}(
                        kkt_solve=:symmetric,
                        iterative_refinement=true,
                        ϵ_tol=1.0e-5,
                        ϵ_al_tol=1.0e-5,
                        max_iter=250,
                        verbose=true,
                        linear_solver=:QDLDL,
                        )

s = NCSolver(x0,model,opts=opts)
@time solve!(s)
