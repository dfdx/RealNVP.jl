# using Yota
include("../../Yota/src/core.jl")
using Distributions

include("primitives.jl")
include("coupling.jl")



# function logprob(c::Coupling, x)
# end


mutable struct RealNVP
    prior::MvNormal
    c1::Coupling
    c2::Coupling
end

Base.show(io::IO, m::RealNVP) = print(io, "RealNVP(..)")


function RealNVP(xz_len::Int, u_len::Int)
    mlps = [MLP3(xz_len, u_len, u_len, xz_len) for i=1:4]
    mask = vcat([1 for i=1:Int(xz_len / 2)], [0 for i=1:Int(xz_len / 2)])
    neg_mask = [1 - x for x in mask]
    prior = MvNormal(zeros(xz_len), ones(xz_len))  # note: not CuArray friendly right now
    RealNVP(prior,
            Coupling(mask, mlps[1], mlps[2]),
            Coupling(neg_mask, mlps[3], mlps[4]))
end

function fwd_map(flow::RealNVP, x)
    z, log_det_J1 = fwd_map(flow.c1, x)
    z, log_det_J2 = fwd_map(flow.c2, z)
    logp = log_det_J1 - log_det_J2
    return z, logp
end


# function Distributions.gradlogpdf(d::MvNormal, x::AbstractMatrix)
    
# end
@diffrule logpdf(d::MvNormal, x) d 0
@diffrule logprob(d::MvNormal, x) x 0 # gradlogpdf(d, x)

function logprob(flow::RealNVP, x)
    z, logp = fwd_map(flow, x)
    return logpdf(flow.prior, z) .+ logp
end


function loss(flow::RealNVP, x)
    return -mean(logprob(flow, x))
end


function main()
    flow = RealNVP(10, 5)
    x = rand(10, 20)
    _, g = grad(loss, flow, x)
end



loss2(prior, x) = sum(logpdf(prior, x))

function aux()
    flow = RealNVP(2, 3)
    prior = flow.prior
    x = rand(2, 1)
    _, tape = trace(loss2, prior, x)
end



function main2()
    mask = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    s = MLP3{Float64}(10, 5, 5, 10)
    t = MLP3{Float64}(10, 5, 5, 10)
    c = Coupling(mask, s, t)

    x = rand(10, 20)
    z, log_det_J = fwd_map(c, x)
    inv_map(c, z) .- x
end
