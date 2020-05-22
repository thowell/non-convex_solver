
function Model(nlp)
    nlp.meta.lcon
    nlp.meta.ucon

    n = nlp.meta.nvar
    m = nlp.meta.ncon

    # println("n: $n\nm: $m")

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
end

# function Ipopt.createProblem(nlp::AbstractNLPModel)
#     function eval_f(x)
#         obj(nlp, x)
#     end
#     function eval_grad_f(x, grad_f)
#         grad!(nlp, x, grad_f)
#     end
#     function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
#         if mode == :Structure
#             hess_structure!(nlp, rows, cols)
#         else
#             hess_coord!(nlp, x, lambda, values, obj_weight=obj_factor)
#         end
#     end
#     function eval_g(x, g)
#         cons!(nlp, x, g)
#     end
#     function eval_jac_g(x, mode, rows, cols, values)
#         if mode == :Structure
#             jac_structure!(nlp, rows, cols)
#         else
#             jac_coord!(nlp, x, values)
#         end
#     end
#
#     n = nlp.meta.nvar
#     m = nlp.meta.ncon
#     x_L = nlp.meta.lvar
#     x_U = nlp.meta.uvar
#     g_L = nlp.meta.lcon
#     g_U = nlp.meta.ucon
#
#     prob = createProblem(n, x_L, x_U, m, g_L, g_U, nlp.meta.nnzj, nlp.meta.nnzh,
#         eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
#     prob.x .= nlp.meta.x0
#     prob
# end
#
# function write_ipopt_options(root_dir = @__DIR__)
#     outfile=joinpath(root_dir,"ipopt.out")
#     optfile=joinpath(root_dir,"ipopt.opt")
#
#     if !isfile(optfile)
#         println("Writing Ipopt options file to $optfile...")
#         f = open(optfile,"w")
#         println(f,"# IPOPT Options for TrajectoryOptimization.jl\n")
#         println(f,"# Use Quasi-Newton methods to avoid the need for the Hessian")
#         println(f,"hessian_approximation limited-memory\n")
#         println(f,"# Output file")
#         println(f,"file_print_level 5")
#         println(f,"output_file"*" "*"\""*"$(outfile)"*"\"")
#         close(f)
#     end
# end
#
# function parse_ipopt_summary(file=joinpath(@__DIR__,"ipopt.out"))
#     props = Dict()
#     obj = Vector{Float64}()
#     c_max = Vector{Float64}()
#     iter_lines = false  # Flag true if it's parsing the iteration summary lines
#
#     if !isfile(file)
#         println("Can't find Ipopt output file.\n Solve statistics will be unavailable.")
#         return props
#     end
#
#
#     function stash_prop(ln::String,prop::String,prop_name::Symbol=prop,vartype::Type=Float64)
#         if occursin(prop,ln)
#             loc = findfirst(prop,ln)
#             if vartype <: Real
#                 val = convert(vartype,parse(Float64,split(ln)[end]))
#                 props[prop_name] = val
#                 return true
#             end
#         end
#         return false
#     end
#
#     function store_itervals(ln::String)
#         if iter_lines
#             vals = split(ln)
#             if length(vals) == 10 && vals[1] != "iter" && vals[1] != "Restoration" && vals[2] != "iteration"
#                 push!(obj, parse(Float64,vals[2]))
#                 push!(c_max, parse(Float64,vals[3]))
#             end
#         end
#     end
#
#
#     open(file) do f
#         for ln in eachline(f)
#             stash_prop(ln,"Number of Iterations..",:iterations,Int64) ? iter_lines = false : nothing
#             stash_prop(ln,"Total CPU secs in IPOPT (w/o function evaluations)",:self_time)
#             stash_prop(ln,"Total CPU secs in NLP function evaluations",:function_time)
#             stash_prop(ln,"Number of objective function evaluations",:objective_calls,Int64)
#             length(ln) > 0 && split(ln)[1] == "iter" && iter_lines == false ? iter_lines = true : nothing
#             store_itervals(ln)
#         end
#     end
#     props[:cost] = obj
#     props[:c_max] = c_max
#     return props
# end
