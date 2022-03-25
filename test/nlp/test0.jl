n = 50
m = 30

x0 = ones(n)

xL = -Inf*ones(n)
xL[1] = -10.
xL[2] = -5.
xU = Inf*ones(n)
xU[5] = 20.

f_func(x) = x'*x
c_func(x) = x[1:m].^2 .- 1.2

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
