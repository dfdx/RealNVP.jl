
mutable struct Coupling
    mask  # {0, 1} array for input splitting
    s     # callable struct representing s function (scale)
    t     # callable struct representing t function (translation)
end


Base.show(io::IO, c::Coupling) = print(io, "Coupling($(size(c.mask)), $(typeof(c.s)), $(typeof(c.t)))")

"""
Forward mapping x → z. Same as f() function in the paper
"""
function fwd_map(c::Coupling, x)
    mask, s, t = c.mask, c.s, c.t
    xp = mask .* x
    sv = s(xp) .* (1 .- mask)
    tv = t(xp) .* (1 .- mask)
    z = (1 .- mask) .* (x .- tv) .* exp.(-sv) .+ xp
    log_det_J = reshape(sum(sv; dims=1), size(sv, 2))
    return z, log_det_J
end


"""
Inverse mapping z → x. Same as g() function in the paper
"""
function inv_map(c::Coupling, z)
    mask, s, t = c.mask, c.s, c.t
    zp = mask .* z
    sv = s(zp) .* (1 .- mask)
    tv = t(zp) .* (1 .- mask)
    x = zp .+ (1 .- mask) .* (z .* exp.(sv) .+ tv)
    return x
end


########################

function c_loss(c::Coupling, x)
    z, log_det_J = fwd_map(c, x)
    return sum(abs2.(x .- z))
end

function main_coupling()
    flow = RealNVP(2, 256)
    c = flow.c1
    X, y = make_moons()
    x = X[:, 1:10]

    _, tape = trace(c_loss, c, x; optimize=false)
    _, g = grad(c_loss, c, x)

    g[1][(:mask,)]
end


function main_coupling2()
    # verified - works
    lr = 1e-5
    X, y = make_moons()
    x = X[:, 1:10]
    flow = RealNVP(2, 256)
    c = flow.c1
    for epoch=1:100
        epoch_cost = 0
        for (i, x) in enumerate(eachbatch(X, size=10))
            cost, g = grad(c_loss, c, x)
            update!(c, g[1], (x, gx) -> x .- lr * gx)
            epoch_cost += cost
        end
        println(epoch_cost)
    end
end
