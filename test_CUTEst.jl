using CUTEst, NLPModels
include("src/interior_point.jl")

# nlp = CUTEstModel("HS35")

nlp = CUTEstModel("BYRDSPHR")

nlp.meta.lcon
nlp.meta.ucon

n = nlp.meta.nvar
m = nlp.meta.ncon

x0 = nlp.meta.x0
xL = nlp.meta.lvar
xU = nlp.meta.uvar

f_func(x) = obj(nlp,x)
function ∇f_func!(∇f,x)
    grad!(nlp,x,∇f)
    return nothing
end
function ∇²f_func!(∇²f,x)
    ∇²f .= hess(nlp,x)
    return nothing
end

function c_func!(c,x)
    cons!(nlp,x,c)
    return nothing
end
function ∇c_func!(∇c,x)
    ∇c .= jac(nlp,x)
    return nothing
end
function ∇²cλ_func!(∇²cλ,x,λ)
    ∇²cλ .= hess(nlp,x,λ) - hess(nlp,x)
    return nothing
end

model = Model(n,m,xL,xU,f,∇f_func!,∇²f_func!,c_func!,∇c_func!,∇²cλ_func!)

s = InteriorPointSolver(x0,model,opts=Options{Float64}(max_iter=1000))
@time solve!(s,verbose=true)

finalize(nlp)
