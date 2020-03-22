include("src/interior_point.jl")

n = 3
m = 2
# x0 = [-2.; 3.; 1.]
x0 = [0.; 3; 5.]

xl = -Inf*ones(n)
xl[2] = 0.
xl[3] = 0.
xu = Inf*ones(n)

f_func(x) = x[1]
c_func(x) = [x[1]^2 - x[2] - 1.0;
             x[1] - x[3] - 0.5]

∇f_func(x) = ForwardDiff.gradient(f_func,x)
∇c_func(x) = ForwardDiff.jacobian(c_func,x)

s = Solver(x0,n,m,xl,xu,f_func,c_func,∇f_func,∇c_func; opts=Options{Float64}())
solve!(s)
s.x

a = (1.,1.)
b = (0.,2.)
c = (0.1,3.)

f = Tuple[]
push!(f,a)
push!(f,b)
push!(f,c)

f



add_to_filter((-10.,0.),s)

function add_to_filter(p,s::Solver)
    f = s.filter
    if isempty(f)
        push!(f,p)
        return nothing
    end

    len = length(f)
    for _p in f
        if p[1] >= _p[1] && p[2] >= _p[2]
            len -= 1
        end
    end

    if length(f) == len
        _f = copy(f)
        empty!(f)
        push!(f,p)
        for _p in _f
            if _p[1] < p[1] || _p[2] < p[2]
                push!(f,_p)
            end
        end
    end
    return nothing
end
