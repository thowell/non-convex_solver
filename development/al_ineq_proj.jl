using LinearAlgebra, ForwardDiff
using Convex, SCS, ECOS

"""
 min x'Px
 st y >= 0
 	Ax - b = y
    C*x = d
"""
nc = 5
n = nc
mc = 2
m = 2*mc
P = Diagonal(rand(n))
A = rand(mc,nc)
b = rand(mc)
C = rand(mc,nc)
d = rand(mc)

"Convex.jl"
x = Variable(nc)
y = Variable(mc)
prob = minimize(quadform(x,P))
prob.constraints += y >= 0
prob.constraints += A*x - b - y == 0
prob.constraints += C*x - d == 0
solve!(prob,SCS.Optimizer)

@show prob.status
@show x.value
@show y.value
prob.optval

"Projection"
function Π(x)
	max.(x,0.0)
end

"Augmented Lagrangian"
f(x) = x[1:n]'*P*x[1:n]
c1(x) = x[n .+ (1:mc)] - Π(x[n .+ (1:mc)])
c2(x) = A*x[1:n] - b - x[n .+ (1:mc)]
c3(x) = C*x[1:n] - d
c(x) = [c1(x);c2(x);c3(x)]
# c1p(x) = x[n .+ (1:mc)] - max.(c1(x),0)
# c2p(x) = c2(x)
# c3p(x) = c3(x)
# cpr(x) = [c1p(x);c2p(x);c3p(x)]
L(x,λ,ρ) = f(x) + λ'*c(x) + 0.5*ρ*c(x)'*c(x)

function solve(x)
	x = copy(x)

	λ = zeros(3mc)
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
		norm(c(x)) < 1.0e-6 && break

		λ[1:mc] = Π(λ[1:mc] + ρ*x[n .+ (1:mc)])
		λ[mc .+ (1:2mc)] = λ[mc .+ (1:2mc)] + ρ*[c2(x);c3(x)]

		ρ *= 10.0

		k += 1
	end

	return x
end

x0 = rand(nc+mc)
x_sol = solve(x0)
@show x_sol

f(x_sol)
f(x.value)
c1(x_sol)
c2(x_sol)
c3(x_sol)
x_sol[n .+ (1:mc)]
y.value
norm(x_sol[1:n] - x.value)
