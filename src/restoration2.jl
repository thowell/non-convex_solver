function restoration!(s::Solver)
    println("restoration phase")

    if !kkt_error_reduction(s)
        error("-KKT error reduction failure")
    else
        println("-KKT error reduction success")
    end
end
