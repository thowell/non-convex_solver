using CUTEst, NLPModels
include("../src/interior_point.jl")


# nlp = CUTEstModel("HS35")
# nlp = CUTEstModel("STRTCHDVNE")
# nlp = CUTEstModel("TRIGON1NE")
# nlp = CUTEstModel("WATER") # inf det.
# nlp = CUTEstModel("ODFITS")
# nlp = CUTEstModel("STRTCHDVNE")
# nlp = CUTEstModel("HS111")
# nlp = CUTEstModel("GENHS28")

# nlp = CUTEstModel("ARGLCLE") ## factorization issue -> reg fix init.
nlp = CUTEstModel("BT1")
# nlp = CUTEstModel("BT2")
# nlp = CUTEstModel("BT3")
# nlp = CUTEstModel("BT4")
# nlp = CUTEstModel("BT5")
# nlp = CUTEstModel("BT6")
# nlp = CUTEstModel("BT7")
# nlp = CUTEstModel("BT8")
# nlp = CUTEstModel("BT9")
# nlp = CUTEstModel("BT10")
# nlp = CUTEstModel("BT11")
# nlp = CUTEstModel("BT12")
# nlp = CUTEstModel("BYRDSPHR")
# nlp = CUTEstModel("COOLHANS")
# nlp = CUTEstModel("DIXCHLNG")
# nlp = CUTEstModel("EIGENA2") ##big
# nlp = CUTEstModel("EIGENACO") ##big
# nlp = CUTEstModel("EIGENB2") ##big
# nlp = CUTEstModel("EIGENBCO") ##big
# nlp = CUTEstModel("EIGENBC2")
# nlp = CUTEstModel("ELEC") ## calls watchdog and restoration
# nlp = CUTEstModel("GRIDNETE") ##big
# nlp = CUTEstModel("GRIDNETH") ## big
# nlp = CUTEstModel("HS6")
# nlp = CUTEstModel("HS7")
# nlp = CUTEstModel("HS8")
# nlp = CUTEstModel("HS9")
# nlp = CUTEstModel("HS26")
# nlp = CUTEstModel("HS27")
# nlp = CUTEstModel("HS28")
# nlp = CUTEstModel("HS39")
# nlp = CUTEstModel("HS40")
# nlp = CUTEstModel("HS42")
# nlp = CUTEstModel("HS46")
# nlp = CUTEstModel("HS47")
# nlp = CUTEstModel("HS48")
# nlp = CUTEstModel("HS49")
# nlp = CUTEstModel("HS50")
# nlp = CUTEstModel("HS51")
# nlp = CUTEstModel("HS52")
# nlp = CUTEstModel("HS56")
# nlp = CUTEstModel("HS61")
# nlp = CUTEstModel("HS77")
# nlp = CUTEstModel("HS78")
# nlp = CUTEstModel("HS79")
# nlp = CUTEstModel("LCH") ##big watchdog
# nlp = CUTEstModel("LUKVLE1") ##big
# nlp = CUTEstModel("LUKVLE3") ##big
# nlp = CUTEstModel("LUKVLE6") ##big
# nlp = CUTEstModel("LUKVLE7") ##big
# nlp = CUTEstModel("LUKVLE8")
# nlp = CUTEstModel("LUKVLE9")
# nlp = CUTEstModel("LUKVLE10")
# nlp = CUTEstModel("LUKVLE13")
# nlp = CUTEstModel("LUKVLE14")
# nlp = CUTEstModel("LUKVLE16")
# nlp = CUTEstModel("MSS1")
# nlp = CUTEstModel("MARATOS")
# nlp = CUTEstModel("MWRIGHT")
# nlp = CUTEstModel("ORTHRDM2") ##big
# nlp = CUTEstModel("ORTHRDS2") ##big
# nlp = CUTEstModel("ORTHREGA") ##big
# nlp = CUTEstModel("ORTHREGB")
# nlp = CUTEstModel("ORTHREGC") ##big
# nlp = CUTEstModel("ORTHREGD") ##big
# nlp = CUTEstModel("S316-322")
# nlp = CUTEstModel("WOODSNE") ##big

nlp.meta.lcon
nlp.meta.ucon

n = nlp.meta.nvar
m = nlp.meta.ncon

println("n: $n\nm: $m")

x0 = nlp.meta.x0
xL = nlp.meta.lvar
xU = nlp.meta.uvar

f_func(x,model) = obj(nlp,x)
function ∇f_func!(∇f,x,model)
    grad!(nlp,x,∇f)
    return nothing
end
function ∇²f_func!(∇²f,x,model)
    ∇²f .= hess(nlp,x)
    return nothing
end

function c_func!(c,x,model)
    cons!(nlp,x,c)
    return nothing
end
function ∇c_func!(∇c,x,model)
    ∇c .= jac(nlp,x)
    return nothing
end
function ∇²cy_func!(∇²cy,x,y,model)
    ∇²cy .= hess(nlp,x,y) - hess(nlp,x)
    return nothing
end

model = Model(n,m,xL,xU,f_func,∇f_func!,∇²f_func!,c_func!,∇c_func!,∇²cy_func!,cA_idx=ones(Bool,m))

opts = Options{Float64}(kkt_solve=:symmetric,
                        iterative_refinement=true,
                        relax_bnds=true,
                        max_iter=1000,
                        y_init_ls=true,
                        verbose=false)

s = InteriorPointSolver(x0,model,opts=opts)
@time solve!(s)

finalize(nlp)
