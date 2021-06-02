using Calculus, ForwardDiff

differentiate("cos(x) + sin(x) + exp(-x) * cos(x)", :x)
differentiate(differentiate("cos(x) + sin(y) + exp(-x) * cos(y)", [:x, :y])[1],[:x,:y])

x = rand(2)
_f(x,y) = cos(x) + sin(y) + exp(-x) * cos(y)
ForwardDiff.hessian(_f,x)
my_f(x,y) = ((0 * -(sin(x)) + 1 * -(1 * cos(x))) + (((0 * exp(-x) + -1 * (-1 * exp(-x))) * cos(y) + (-1 * exp(-x)) * 0) + ((-1 * exp(-x)) * 0 + exp(-x) * 0)))
