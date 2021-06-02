using LinearAlgebra, ForwardDiff
using Convex, SCS, ECOS

"""
 min x'Px
 st Ax <= b
    C*x = d
"""

n = 5
mc = 2
m = 2*mc
P = Diagonal(rand(n))
A = rand(mc,n)
b = rand(mc)
C = rand(mc,n)
d = rand(mc)

"Convex.jl"
x = Variable(n)
prob = minimize(quadform(x,P))
prob.constraints += A*x <= b
prob.constraints += C*x - d == 0
solve!(prob,SCS.Optimizer)

@show prob.status
@show x.value
prob.optval

"Augmented Lagrangian"
f(x) = x'*P*x
c1(x) = A*x - b
c2(x) = C*x - d
c(x) = [c1(x);c2(x)]
c1p(x) = max.(c1(x),0)
c2p(x) = c2(x)
cpr(x) = [c1p(x);c2p(x)]
L(x,λ,ρ) = f(x) + λ'*cpr(x) + 0.5*ρ*cpr(x)'*cpr(x)

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
		norm(cpr(x)) < 1.0e-6 && break

		λ[1:mc] = max.(λ[1:mc] + ρ*(A*x - b),0.0)
		λ[mc .+ (1:mc)] = λ[mc .+ (1:mc)] + ρ*c2(x)

		ρ *= 10.0

		k += 1
	end

	return x
end

x0 = randn(n)
x_sol = solve(x0)
@show x_sol

norm(x_sol - x.value)
