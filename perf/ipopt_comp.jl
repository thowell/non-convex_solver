using BenchmarkTools
using Statistics
using Plots
include("../test/cutest.jl")
include("../test/test_utils.jl")

write_ipopt_options(@__DIR__)

perf_ipopt = BenchmarkTools.Trial[]
perf_ip = BenchmarkTools.Trial[]
status = Symbol[]
iters = Tuple{Int,Int}[]
@progress for prob in CUTEstProbs[:small]
    nlp = CUTEstModel(prob)
    tol = 1e-8
    model = Model(nlp)
    opts = Options{Float64}(kkt_solve=:symmetric,
                            iterative_refinement=true,
                            relax_bnds=true,
                            max_iter=100,
                            y_init_ls=true,
                            max_iterative_refinement=1000,
                            ϵ_tol = 1e-8,
                            verbose=false)
    solver = InteriorPointSolver(nlp.meta.x0, model, cA_idx=zeros(Bool,nlp.meta.ncon),opts=opts)
    x0 = copy(nlp.meta.x0)

    ipoptProb = createProblem(nlp)
    addOption(ipoptProb, "print_level", 0)
    optfile = joinpath(@__DIR__, "ipopt.opt")
    addOption(ipoptProb, "option_file_name", optfile)
    st = 0
    b_ipopt = @benchmark begin
        $ipoptProb.x .= $x0
        $st = solveProblem($ipoptProb)
    end samples=10 evals=10
    stats = parse_ipopt_summary(joinpath(@__DIR__,"ipopt.out"))

    try
        b_ip = benchmark_solve!(solver, samples=10, evals=10)
        if norm(solver.s.x - ipoptProb.x) < 1e-6
            push!(status, :match)
            push!(perf_ip, b_ip)
            push!(perf_ipopt, b_ipopt)
        else
            push!(status, :nomatch)
        end
    catch
        if st == 0
            push!(status, :ip_failed)
        else
            push!(status, :both_failed)
        end
    end
    push!(iters,(solver.s.k, stats[:iterations]))
    finalize(nlp)
end
count(status .== :match)/length(status)
count(status .!= :ip_failed)/length(status)

jmt = map(zip(perf_ip, perf_ipopt)) do (b1,b2)
    judge(minimum(b1), minimum(b2))
end
tr = [j.ratio.time for j in jmt]
mr = [j.ratio.memory for j in jmt]
count(tr .< 1)/length(tr)
count(mr .< 1)/length(tr)

mean([j.ratio.time for j in jmt])
mean([j.ratio.memory for j in jmt])
