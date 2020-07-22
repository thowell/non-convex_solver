using LinearAlgebra, ForwardDiff
using Convex, SCS, ECOS

"""
 min x'Px
 st Ax = b
"""

n = 5
m = 2
P = Diagonal(rand(n))
A = rand(m,n)
b = rand(m)

"Convex.jl"
x = Variable(n)
prob = minimize(quadform(x,P))
prob.constraints += A*x-b == 0
solve!(prob,SCS.Optimizer)

@show prob.status
@show x.value
prob.optval

"Augmented Lagrangian"
f(x) = x'*P*x
c(x) = A*x - b
L(x,λ,ρ) = f(x) + λ'*c(x) + 0.5*ρ*c(x)'*c(x)

function solve(x)
	x = copy(x)

	λ = zeros(m)
	ρ = 1.0

	k = 1
	while k < 10
		i = 1
		while i < 10
			_L(z) = L(z,λ,ρ)
			f = _L(x)
			∇L = ForwardDiff.gradient(_L,x)
			norm(∇L) < 1.0e-5 && break
			∇²L = ForwardDiff.hessian(_L,x)
			Δx = -(∇²L + 1.0e-5*I)\∇L

			α = 1.0

			j = 1
			while j < 10
				if (_L(x + α*Δx) <= f + 1.0e-4*α*Δx'*∇L) && (-Δx'*ForwardDiff.gradient(_L,x+α*Δx) <= -0.9*Δx'*∇L)
					break
				else
					α *= 0.5
					j += 1
				end
			end
			j == 10 && @warn "line search failed"

			x .+= α*Δx
			i += 1
		end
		norm(c(x)) < 1.0e-5 && break

		λ += ρ*c(x)
		ρ *= 10.0

		k += 1
	end

	return x
end

x0 = rand(n)
x_sol = solve(x0)
@show x_sol

norm(x_sol - x.value)
