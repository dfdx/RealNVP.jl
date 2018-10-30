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

MLP3(n1, n2, n3, n4) = MLP3{Float64}(n1, n2, n3, n4)


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



#         log_det_J, z = x.new_zeros(x.shape[0]), x
#         for i in reversed(range(len(self.t))):
#             z_ = self.mask[i] * z
#             s = self.s[i](z_) * (1-self.mask[i])
#             t = self.t[i](z_) * (1-self.mask[i])
#             z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
#             log_det_J -= s.sum(dim=1)
#         return z, log_det_J

function f_func(c::Coupling, x)
    mask, s, t = c.mask, c.s, c.t
    xp = mask .* x
    sv = s(xp) .* (1 .- mask)
    tv = t(xp) .* (1 .- mask)
    z = (1 .- mask) .* (x .- tv) .* exp.(-sv) .+ xp
    return z
end

# function f_func(c::Coupling, x)
#     mask, s, t = c.mask, c.s, c.t
#     xp, xn = mask .* x, (1 .- mask) .* x
#     # return xp .+ xn .* exp.(s(xp)) .+ t(xp)   # 1st formula
#     return xp .+ (1 .- mask) .* (x .* exp.(s(xp)) .+ t(xp)) # 2nd formula
# end




        # x = z
        # for i in range(len(self.t)):
        #     x_ = x*self.mask[i]
        #     s = self.s[i](x_)*(1 - self.mask[i])
        #     t = self.t[i](x_)*(1 - self.mask[i])
        #     x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
# return x


function g_func(c::Coupling, z)
    mask, s, t = c.mask, c.s, c.t
    zp = mask .* z
    sv = s(zp) .* (1 .- mask)
    tv = t(zp) .* (1 .- mask)
    x = zp .+ (1 .- mask) .* (z .* exp.(sv) .+ tv)
    return x
end


# function g_func(c::Coupling, z)
    # mask, s, t = c.mask, c.s, c.t
    # zp, zn = mask .* z, (1 .- mask) .* z  # z positive, z negative
    # return zp .+ (zn - t(zp)) .* exp.(-s(zp))
# end




function logprob(c::Coupling, x)
end



function main2()
    mask = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    s = MLP3{Float64}(10, 5, 5, 10)
    t = MLP3{Float64}(10, 5, 5, 10)
    c = Coupling(mask, s, t)
    
    x = rand(10, 20)
    z = f_func(c, x)
    g_func(c, z) .- x
end
