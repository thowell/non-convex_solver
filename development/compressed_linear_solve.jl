using QDLDL, BenchmarkTools

s.s.opts.iterative_refinement = false
@benchmark search_direction_fullspace!($s.s)
@benchmark search_direction_symmetric!($s.s)
@benchmark search_direction_slack!($s.s)


inertia_correction_slack!(s)
s = s.s

function test_1(s::Solver)
    n = s.model_opt.n
    m = s.model_opt.m
    mI = s.model.mI
    mE = s.model.mE
    mA = s.model.mA
    nL = s.model_opt.nL
    nU = s.model_opt.nU
    idx = s.idx

    ΔxL = view(s.ΔxL,1:nL)
    ΔsL = view(s.ΔxL,nL .+ (1:mI))
    ΔxU = s.ΔxU
    zL = view(s.zL,1:nL)
    zS = view(s.zL,nL .+ (1:mI))
    zU = s.zU
    s._dxy .= ma57_solve(s.LBL_slack,-s.h_slack)

    s.dr .= 1.0/(s.ρ + s.δw)*(s.dyA - s.hr)
    s._dzL .= -(zL.*view(s.dxL,1:nL) + s.hzL)./(ΔxL .- s.δc)
    s.dzU .= (zU.*view(s.dx,1:nU) - s.hzU)./(ΔxU .- s.δc)


    s.dzs .= -s.dyI + s.hs
    s.ds .= -((ΔsL .- s.δc).*s.dzs + s.hzs)./zS

    return nothing
end
@benchmark test_1($s)

function test_2(s)
    s.dxy .= ma57_solve(s.LBL, -s.h_sym)
    s.dzL .= -s.σL.*s.dxL - s.zL + s.μ./(s.ΔxL .- s.δc)
    s.dzU .= s.σU.*s.dxU - s.zU + s.μ./(s.ΔxU .- s.δc)
    return nothing
end
@benchmark test_2($s)
@benchmark ma57_factorize($s.LBL)

tmp_mat = s.H + Diagonal(s.δ)
function test_3(s,tmp_mat)
    s.d .= tmp_mat\(-s.h)
    return nothing
end
@benchmark test_3($s,$tmp_mat)

kkt_hessian_symmetric!(s)
tmp_mat2 = s.H_sym + Diagonal(view(s.δ,s.idx.xy))
@benchmark qdldl($tmp_mat2)
F = qdldl(tmp_mat2)

function test_4(s,F)
    s.dxy .= solve(F,-s.h_sym)
    s.dzL .= -s.σL.*s.dxL - s.zL + s.μ./(s.ΔxL .- s.δc)
    s.dzU .= s.σU.*s.dxU - s.zU + s.μ./(s.ΔxU .- s.δc)
    return nothing
end

function test_5(s,F)
    s.dxy .= solve(F,-s.h_sym)
    s.dzL .= -s.σL.*s.dxL - s.zL + s.μ./(s.ΔxL .- s.δc)
    s.dzU .= s.σU.*s.dxU - s.zU + s.μ./(s.ΔxU .- s.δc)
    return nothing
end

@benchmark test_4($s,$F)

d_tmp = solve(F,-s.h_sym)
d_tmp2 = ma57_solve(s.LBL, -s.h_sym)

norm(d_tmp + s.h_sym)
norm(d_tmp - d_tmp2)

s = s.s
H_copy = copy(s.H)
view(H_copy,CartesianIndex.(s.idx.zL,s.idx.xL)) .= -1.0
view(H_copy,CartesianIndex.(s.idx.zU,s.idx.xU)) .= 1.0
view(H_copy,CartesianIndex.(s.idx.zL,s.idx.zL)) .= -s.ΔxL./s.zL
view(H_copy,CartesianIndex.(s.idx.zU,s.idx.zU)) .= -s.ΔxU./s.zU

qdldl(H_copy)
h_copy = copy(s.h)
h_copy[s.idx.zL] .= -s.ΔxL + s.μ./s.zL
h_copy[s.idx.zU] .= -s.ΔxU + s.μ./s.zU

HH_tmp = H_copy + Diagonal(s.δ)
d_tmp1 = (H_copy + Diagonal(s.δ))\(-1*h_copy)
search_direction_fullspace!(s)
norm(d_tmp1 - s.d)

F1 = qdldl(HH_tmp)
@benchmark qdldl($HH_tmp)
d_tmp2 = solve(F1,-h_copy)
@benchmark solve($F1,$(-h_copy))
norm(d_tmp2 - s.d)

F1
F1.workspace

s.model.n
s.s.δ[s.s.idx.x] .= 0.01
F = qdldl(s.s.H_sym+ Diagonal(view(s.s.δ,s.s.idx.xy)))
