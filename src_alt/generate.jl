function generate_gradients(func::Function, num_variables::Int, mode::Symbol)
    @variables x[1:num_variables]

    if mode == :scalar 
        f = func(x)
        fx = Symbolics.gradient(f, x) 
        fxx = Symbolics.jacobian(fx, x) 

        f_func = eval(Symbolics.build_function(f, x))
        fx_func = eval(Symbolics.build_function(fx, x)[2])
        fxx_func = eval(Symbolics.build_function(fxx, x)[2])

        return f_func, fx_func, fxx_func
    elseif mode == :vector 
        f = func(x)
        fx = Symbolics.jacobian(f, x)
        dim = length(f)
        @variables y[1:dim]
        fyxx = Symbolics.hessian(dot(f, y), x) 

        f_func = eval(Symbolics.build_function(f, x)[2])
        fx_func = eval(Symbolics.build_function(fx, x)[2])
        fyxx_func = eval(Symbolics.build_function(fyxx, x, y)[2])

        return f_func, fx_func, fyxx_func
    end
end

