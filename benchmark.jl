using BenchmarkTools

n = 1000
m = 10
H = spzeros(n,n)
A = rand(m,m)
v = view(H,1:m,1:m)

h = zeros(n)
idx = 1:m
y = rand(m)
typeof(idx)
function f1(h::Vector{T},y::Vector{T},idx::UnitRange{Int}) where T
    for i = 1:1000
        h[idx] = y
    end
    return nothing
end
function func1(H::SparseMatrixCSC{T,Int},A::Array{T,2}) where T
    for i = 1:1000
        H[1:m,1:m] .= A
    end
    return nothing
end
function func2(H::SparseMatrixCSC{T,Int},A::Array{T,2}) where T
    for i = 1:1000
        view(H,1:m,1:m) .= A
    end
    return nothing
end
function func3(v::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false},A::Array{T,2}) where T
    for i = 1:1000
        v .= A
    end
    return nothing
end
@benchmark f1($h,$y,$idx)
@benchmark func1($H,$A)
@benchmark func2($H,$A)
@benchmark func3($v,$A)
