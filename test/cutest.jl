using CUTEst
using CSV
using FileIO

function sort_problems(probs;small_size=100,medium_size=500)
    # Categorize the problems
    small = String[]
    medium = String[]
    large = String[]
    sizes = Tuple{Int,Int}[]
    for prob in probs
        try
            nlp = CUTEstModel(prob)
            n,m = nlp.meta.nvar, nlp.meta.ncon
            if nlp.meta.lcon == nlp.meta.ucon
                if n < small_size && m < small_size
                    push!(small, prob)
                elseif n < medium_size && m < medium_size
                    push!(medium, prob)
                else
                    push!(large, prob)
                end
            else
                @warn "prob: $(prob) is not correct form"
            end
            push!(sizes, (n,m))
            finalize(nlp)
        catch
            @warn "prob: $(prob) is not in SIF"
        end
    end
    probs_sorted = Dict(:all => probs, :small => small, :medium => medium,
        :large => large, :sizes => sizes)
end

function write_problems(key)
    open(joinpath(pwd(),"test/cutest_problems_$(string(key)).csv"),"w") do io
        write(io,"$(string(key))=[\n")
        for p in probs_sorted[key]
            write(io,""""$p",\n""")
        end
        write(io,"]")
    end
end

# run if new problems are added to cutest_problems.csv
# path = joinpath(pwd(),"test/cutest_problems.csv")
# probs = CSV.read(path)[!,"name"]
# probs_sorted = sort_problems(unique(probs))
#
# write_problems(:small)
# write_problems(:medium)
# write_problems(:large)
