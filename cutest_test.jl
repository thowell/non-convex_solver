using CUTEst, NLPModels
include("src/interior_point.jl")

finalize(nlp)
nlp = CUTEstModel("ROBOTARM")

nlp.meta.lcon
nlp.meta.ucon

n = nlp.meta.nvar
m = nlp.meta.ncon

x0 = nlp.meta.x0
xL = nlp.meta.lvar
xU = nlp.meta.uvar

f_func(x) = obj(nlp,x)
c_func(x) = cons(nlp,x)

∇f_func(x) = grad(nlp,x)
∇c_func(x) = jac(nlp,x)

∇²L_func(x,λ) = hess(nlp,x,λ)

s = InteriorPointSolver(x0,n,m,xL,xU,f_func,c_func,∇f_func,∇c_func,∇²L_func,opts=Options{Float64}(max_iter=1000))
@time solve!(s,verbose=true)
