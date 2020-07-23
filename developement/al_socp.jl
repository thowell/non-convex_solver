using LinearAlgebra, ForwardDiff
using Convex, SCS, ECOS

"""
 min x'Px
 st Ax = b
    ||x|| <= t
"""

n = 5
mc = 2
m = mc
P = Diagonal(rand(n))
A = rand(mc,n)
b = rand(mc)
t = 10.0*rand(1)[1]

"Convex.jl"
x = Variable(n)
prob = minimize(quadform(x,P))
prob.constraints += A*x == b
prob.constraints += norm(x) <= t
solve!(prob,SCS.Optimizer)

@show prob.status
@show x.value
prob.optval

"Second-order cone projection"
function Π(v,s)
	if norm(v) <= -s
		# @warn "below cone"
		return zero(v), 0.0
	elseif norm(v) <= s
		# @warn "in cone"
		return v, s
	elseif norm(v) > abs(s)
		# @warn "outside cone"
		a = 0.5*(1.0 + s/norm(v))
		return a*v, a*norm(v)
	else
		@warn "soc projection error"
		return zero(v), 0.0
	end
end

"Augmented Lagrangian"
f(x) = x'*P*x
c1(x) = A*x - b
function c2(x)
	xp, tp = Π(x,t)
	[x;t] - [xp;tp]
end
c(x) = [c1(x);c2(x)]

L(x,λ,ρ) = f(x) + λ'*c(x) + 0.5*ρ*c(x)'*c(x)

function solve(x)
	x = copy(x)

	λ = zeros(m+n+1)
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

		λ[1:m] += λ[1:m] + ρ*c1(x)

		tmp = λ[m .+ (1:n+1)] + ρ*([x;t])
		λxp, λtp = Π(tmp[1:n],tmp[n+1])
		λ[m .+ (1:n+1)] = [λxp;λtp]
		# λ[m .+ (1:n+1)] = λ[m .+ (1:n+1)] + ρ*c2(x)

		ρ *= 10.0

		k += 1
	end

	return x, λ, ρ
end

x0 = randn(n)
x_sol, λ_sol, ρ_sol = solve(x0)
@show x_sol

norm(x_sol - x.value)
