function problem1()
    n = 50
    m = 30

    x0 = ones(n)

    xL = -Inf*ones(n)
    xL[1] = -10.
    xL[2] = -5.
    xU = Inf*ones(n)
    xU[5] = 20.

    f_func(x) = x'*x
    f, ∇f!, ∇²f! = objective_functions(f_func)

    c_func(x) = x[1:m].^2 .- 1.2
    c!, ∇c!, ∇²cy! = constraint_functions(c_func)

    model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=zeros(Bool,m),cA_idx=ones(Bool,m))

    return x0, model
end

function wachter()
    n = 3
    m = 2

    x0 = [-2.0;3.0;1.0]

    xL = -Inf*ones(n)
    xL[2] = 0.
    xL[3] = 0.
    xU = Inf*ones(n)

    f_func(x) = x[1]
    f, ∇f!, ∇²f! = objective_functions(f_func)

    c_func(x) = [x[1]^2 - x[2] - 1.0;
                 x[1] - x[3] - 0.5]
    c!, ∇c!, ∇²cy! = constraint_functions(c_func)

    model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cA_idx=ones(Bool,m))

    return x0, model
end

function wachter_reform()
    n = 1
    m = 2

    x0 = [-2.0]

    xL = -Inf*ones(n)
    xU = Inf*ones(n)

    f_func(x) = x[1]
    f, ∇f!, ∇²f! = objective_functions(f_func)

    c_func(x) = [x[1]^2 - 1.0;
                 x[1] - 0.5]
    c!, ∇c!, ∇²cy! = constraint_functions(c_func)

    cI_idx=ones(Bool,m)
    cA_idx=ones(Bool,m)
    model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=cI_idx,cA_idx=cA_idx)

    return x0, model
end

function problem2()
    n = 2
    m = 0
    x0 = [1.; 0.1]

    xL = -Inf*ones(n)
    xU = [0.25; 0.25]

    f_func(x) = (x[1] + 0.5)^2 + (x[2] - 0.5)^2
    f, ∇f!, ∇²f! = objective_functions(f_func)

    function c_func!(c,x,model::AbstractModel)
        return nothing
    end
    function ∇c_func!(∇c,x,model::AbstractModel)
        return nothing
    end

    function ∇²cy_func!(∇²c,x,y,model::AbstractModel)
        return nothing
    end

    model = Model(n,m,xL,xU,f,∇f!,∇²f!,c_func!,∇c_func!,∇²cy_func!)

    return x0, model
end

function problem3()
    n = 2
    m = 1
    x0 = [-2.; 10.]

    xL = -Inf*ones(n)
    xU = Inf*ones(n)

    f_func(x) = 2.0*(x[1]^2 + x[2]^2 - 1.0) - x[1]
    f, ∇f!, ∇²f! = objective_functions(f_func)

    c_func(x) = [x[1]^2 + x[2]^2 - 1.0]
    c_func_d(x) = x[1]^2 + x[2]^2 - 1.0
    function c_func!(c,x,model)
        c .= c_func(x)
        return nothing
    end
    ∇c_func(x) = Array(ForwardDiff.gradient(c_func_d,x)')

    function ∇²cy_func(x,y)
        ∇cy(x) = ∇c_func(x)'*y
        return ForwardDiff.jacobian(∇cy,x)
    end

    function ∇c_func!(∇c,x,model)
        ∇c .= ∇c_func(x)
        return nothing
    end

    function ∇²cy_func!(∇²cy,x,y,model)
        ∇²cy .= ∇²cy_func(x,y)
        return nothing
    end

    model = Model(n,m,xL,xU,f,∇f!,∇²f!,c_func!,∇c_func!,∇²cy_func!)

    return x0, model
end

function problem4()
    n = 50
    m = 10

    x0 = rand(n+m)

    xL = -Inf*ones(n+m)
    xL[n .+ (1:m)] .= 0.
    xU = Inf*ones(n+m)

    f_func(x) = x'*x
    f, ∇f!, ∇²f! = objective_functions(f_func)

    c_func(x) = 5.0*x[1:m].^2 .- 3.0 - x[n .+ (1:m)]
    c!, ∇c!, ∇²cy! = constraint_functions(c_func)

    model = Model(n+m,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!)

    return x0, model
end

function problem6()
    n = 2
    m = 2

    x0 = rand(n)

    xL = -Inf*ones(n)
    xU = Inf*ones(n)

    f_func(x) = -x[1]*x[2] + 2/(3*sqrt(3))
    f, ∇f!, ∇²f! = objective_functions(f_func)

    c_func(x) = [-x[1] - x[2]^2 + 1.0;
                 x[1] + x[2]]
    c!, ∇c!, ∇²cy! = constraint_functions(c_func)

    model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=ones(Bool,m))

    return x0, model
end

function problem7()
    n = 2
    m = 2

    x0 = rand(n)

    xL = -Inf*ones(n)
    xU = Inf*ones(n)

    f_func(x) = 100*(x[2]-x[1]^2)^2 + (1-x[1])^2
    f, ∇f!, ∇²f! = objective_functions(f_func)

    c_func(x) = [-(x[1] -1)^3 + x[2] - 1;
                 -x[1] - x[2] + 2]
    c!, ∇c!, ∇²cy! = constraint_functions(c_func)

    model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=ones(Bool,m))

    return x0, model
end

function problem8()
    n = 3
    m = 1

    x0 = rand(n)

    xL = -Inf*ones(n)
    xU = Inf*ones(n)

    f_func(x) = x[1] - 2*x[2] + x[3] + sqrt(6)
    f, ∇f!, ∇²f! = objective_functions(f_func)

    c_func(x) = [1 - x[1]^2 - x[2]^2 - x[3]^2]
    c!, ∇c!, ∇²cy! = constraint_functions(c_func)

    model = Model(n,m,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=ones(Bool,m))

    return x0, model
end
