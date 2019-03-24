using Yota

include("primitives.jl")
include("coupling.jl")



# function g_func(c::Coupling, z)
    # mask, s, t = c.mask, c.s, c.t
    # zp, zn = mask .* z, (1 .- mask) .* z  # z positive, z negative
    # return zp .+ (zn - t(zp)) .* exp.(-s(zp))
# end




function logprob(c::Coupling, x)
end

# fun(c, x) = sum(g_func(c, f_func(c, x)))


mutable struct RealVNP
    c1::Coupling
    c2::Coupling
end

function RealNVP(xz_len, u_len)
    mlps = [MLP3(xz_len, u_len, u_len, xz_len) for i=1:4]
    mask = vcat([1 for i=1:Int(xz_len / 2)], [0 for i=1:Int(xz_len / 2)])
    RealNVP(Coupling(mlps[1], mlps[2]),
            Coupling(mlps[3], mlps[4]))
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
