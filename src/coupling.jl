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

MLP3(n1, n2, n3, n4) = MLP3{Float64}(n1, n2, n3, n4)


function Base.show(io::IO, m::MLP3{T}) where T
    n2, n1 = size(m.W1)
    n4, n3 = size(m.W3)
    print(io, "MLP{$T}($n1, $n2, $n3, $n4)")
end


function (m::MLP3{T})(x) where T
    x = relu(m.W1 * x .+ m.b1)
    x = relu(m.W2 * x .+ m.b2)
    x = relu(m.W3 * x .+ m.b3)
    return x
end


loss(m::MLP3, x) = sum(m(x))


function main()
    m = MLP3{Float32}(5, 3, 3, 5)
    x = rand(5, 10)
    v, g = grad(loss, m, x)
end


mutable struct Coupling
    mask  # {0, 1} array for input splitting
    s     # callable struct representing s function (scale)
    t     # callable struct representing t function (translation)
end


function f_func(c::Coupling, x)
    mask, s, t = c.mask, c.s, c.t
    xp = mask .* x
    sv = s(xp) .* (1 .- mask)
    tv = t(xp) .* (1 .- mask)
    z = (1 .- mask) .* (x .- tv) .* exp.(-sv) .+ xp
    return z
end


function g_func(c::Coupling, z)
    mask, s, t = c.mask, c.s, c.t
    zp = mask .* z
    sv = s(zp) .* (1 .- mask)
    tv = t(zp) .* (1 .- mask)
    x = zp .+ (1 .- mask) .* (z .* exp.(sv) .+ tv)
    return x
end
