n = 2
m = 1

x0 = [2.0; 1.0]

xL = -Inf*ones(n)
xU = Inf*ones(n)

f_func(x) = 2.0 * (x[1]^2 + x[2]^2 - 1.0) - x[1]
c_func(x) = [x[1]^2 + x[2]^2 - 1.0]

model = Model(n,m,xL,xU,f_func,c_func,cI_idx=zeros(Bool,m),cA_idx=ones(Bool,m))

opts = Options{Float64}(
                        kkt_solve=:symmetric,
                        iterative_refinement=false,
                        ϵ_tol=1.0e-5,
                        ϵ_al_tol=1.0e-5,
                        max_iter=250,
                        verbose=true,
                        linear_solver=:QDLDL,
                        )

s = NCSolver(x0,model,opts=opts)
@time solve!(s)
