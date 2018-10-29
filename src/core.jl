using Yota

include("primitives.jl")

mutable struct MLP3{T}
    W1::AbstractMatrix{T}
    b1::AbstractVector{T}
    W2::AbstractMatrix{T}
    b2::AbstractVector{T}
    W3::AbstractMatrix{T}
    b3::AbstractVector{T}
end

MLP3{T}(n1, n2, n3, n4) where T = MLP3(randn(T, n2, n1), randn(T, n2),
                                       randn(T, n3, n2), randn(T, n3),
                                       randn(T, n4, n3), randn(T, n4))

function Base.show(io::IO, m::MLP3{T}) where T
    n2, n1 = size(m.W1)
    n4, n3 = size(m.W3)
    print(io, "MLP{$T}($n1, $n2, $n3, $n4)")
end


# function forward(m::MLP3{T}, x) where T
#     x = relu(m.W1 * x .+ m.b1)
#     x = relu(m.W2 * x .+ m.b2)
#     return x
# end


function (m::MLP3{T})(x) where T
    x = relu(m.W1 * x .+ m.b1)
    x = relu(m.W2 * x .+ m.b2)
    return x
end



loss(m::MLP3, x) = sum(m(x))


function main()
    m = MLP3{Float32}(5, 3, 3, 5)
    x = rand(5, 10)
    v, g = grad(loss, m, x)
end


mutable struct Coupling
    s  # callable object
    t  # callable object
end


function f_func(c::Coupling, x)
end


function g_func(c::Coupling, z)
end



function logprob(c::Coupling, x)
end
