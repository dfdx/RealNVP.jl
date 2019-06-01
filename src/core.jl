# based on: https://github.com/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb

# using Yota
# import Yota: trace
include("../../Yota/src/core.jl")
using Distributions
using MLDataUtils
using StatsBase
using GradDescent
using MLDataUtils
using MLDatasets
using StatsBase
using ImageView
using Plots

gr()


include("primitives.jl")
include("coupling.jl")
include("utils.jl")


mutable struct RealNVP
    prior::MvNormal
    c1::Coupling
    c2::Coupling
end

Base.show(io::IO, m::RealNVP) = print(io, "RealNVP(.)")


function RealNVP(xz_len::Int, u_len::Int)
    mlps = [MLP3(xz_len, u_len, u_len, xz_len) for i=1:4]
    mask = vcat([1 for i=1:Int(xz_len / 2)], [0 for i=1:Int(xz_len / 2)])
    neg_mask = [1 - x for x in mask]
    prior = MvNormal(zeros(xz_len), ones(xz_len))  # note: not CuArray friendly right now
    RealNVP(prior,
            Coupling(mask, mlps[1], mlps[2]),
            Coupling(neg_mask, mlps[3], mlps[4]))
end

# TODO: don't update mask (new parameter in update!?)
# TODO: also mask derivative seems to have wrong size, resulting in wrong update!()
function fwd_map(flow::RealNVP, x)
    z, log_det_J1 = fwd_map(flow.c1, x)
    z, log_det_J2 = fwd_map(flow.c2, z)
    logp = log_det_J1 - log_det_J2
    return z, logp
end


function Distributions.gradlogpdf(ds::AbstractVector, d::MvNormal, x::AbstractMatrix)
    ret = similar(x)
    for j in 1:size(x, 2)
        ret[:, j] = gradlogpdf(d, @view x[:, j]) .* ds[j]
    end
    return ret
end

@nodiff Distributions.logpdf(_d::MvNormal, _x) _d
@diffrule Distributions.logpdf(_d::MvNormal, _x) _x Distributions.gradlogpdf(ds, _d, _x)

function logprob(flow::RealNVP, x)
    z, logp = fwd_map(flow, x)
    return Distributions.logpdf(flow.prior, z) .+ logp
end


function loss(flow::RealNVP, x)
    return -mean(logprob(flow, x))
end

function loss2(flow::RealNVP, x)
    return -mean(fwd_map(flow, x).logp)
end


function StatsBase.fit!(flow::RealNVP, X::AbstractMatrix{T};
              n_epochs=50, batch_size=100, lr=1e-7) where T
    for epoch in 1:n_epochs
        print("Epoch $epoch: ")
        epoch_cost = 0
        t = @elapsed for (i, x) in enumerate(eachbatch(X, size=batch_size))
            cost, g = grad(loss, flow, x)
            update!(flow, g[1], (x, gx) -> x .- lr * gx)
            epoch_cost += cost
        end
        println("avg_cost=$(epoch_cost / (size(X,2) / batch_size)), elapsed=$t")
    end
    return flow
end


# function reconstruct(m::RealNVP, x::AbstractVector)
#     x = reshape(x, length(x), 1)
#     mu, _ = encode(m, x)
#     z = mu
#     x_rec = decode(m, z)
#     return x_rec
# end


function show_pic(x)
    reshape(x, 28, 28)' |> imshow
end


function show_recon(m, x)
    x_ = reconstruct(m, x)
    show_pic(x)
    show_pic(x_)
end


function main()
    # TODO: simplify coupling layers for stability or use smaller dataset
    # TODO: use make_moons() first, draw X as zeros
    # X, _ = MNIST.traindata()
    # X = convert(Matrix{Float64}, reshape(X, 784, 60000))
    X, y = make_moons()
    scatter(X[1, :], X[2, :], color=y)


    flow = RealNVP(2, 256)
    @time fit!(flow, X; batch_size=10, n_epochs=1000, lr=1e-8)

    # check reconstructed image
    for i=1:2:10
        show_recon(flow, X[:, i])
    end
end



