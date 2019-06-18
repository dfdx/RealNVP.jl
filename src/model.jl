mutable struct RealNVP
    prior::MvNormal
    c1::Coupling
    c2::Coupling
end

Base.show(io::IO, m::RealNVP) = print(io, "RealNVP($(m.c1), $(m.c2))")


function RealNVP(xz_len::Int, u_len::Int)
    mask = vcat([1 for i=1:Int(xz_len / 2)], [0 for i=1:Int(xz_len / 2)])
    neg_mask = [1 - x for x in mask]
    prior = MvNormal(zeros(xz_len), ones(xz_len))  # note: not CuArray friendly right now
    c1 = Coupling(mask, MLP3Tanh(xz_len, u_len, xz_len), MLP3(xz_len, u_len, xz_len))
    c2 = Coupling(neg_mask, MLP3Tanh(xz_len, u_len, xz_len), MLP3(xz_len, u_len, xz_len))
    RealNVP(prior, c1, c2)
end


function fwd_map(flow::RealNVP, x)
    z, log_det_J1 = fwd_map(flow.c1, x)
    z, log_det_J2 = fwd_map(flow.c2, z)
    logp = -log_det_J1 - log_det_J2
    return z, logp
end


function inv_map(flow::RealNVP, z)
    x = inv_map(flow.c2, z)
    x = inv_map(flow.c1, x)
    return x
end


function Distributions.gradlogpdf(ds::AbstractVector, d::MvNormal, x::AbstractMatrix)
    ret = similar(x)
    for j in 1:size(x, 2)
        ret[:, j] = Distributions.gradlogpdf(d, @view x[:, j]) .* ds[j]
    end
    return ret
end

@nodiff Distributions.logpdf(_d::MvNormal, _x) _d
@diffrule Distributions.logpdf(_d::MvNormal, _x) _x Distributions.gradlogpdf(ds, _d, _x)


function logprob(flow::RealNVP, x)
    z, logp = fwd_map(flow, x)
    # return Distributions.logpdf(flow.prior, z) .+ logp
    return Distributions.logpdf(flow.prior, z)
    # return logp
end


function loss(flow::RealNVP, x)
    return -mean(logprob(flow, x))
end


function StatsBase.fit!(flow::RealNVP, X::AbstractMatrix{T};
                        n_epochs=50, batch_size=100, lr=1e-7, report_every=1) where T    
    for epoch in 1:n_epochs
        epoch_cost = 0
        t = @elapsed for (i, x) in enumerate(eachbatch(X, size=batch_size))
            cost, g = grad(loss, flow, x)
            update!(flow, g[1], (x, gx) -> x .- GradDescent.update(Adam(α=lr), gx);
                    ignore=[(:c1, :mask), (:c2, :mask)])
            epoch_cost += cost
        end
        if epoch % report_every == 0
            println("Epoch $epoch: avg_cost=$(epoch_cost / (size(X,2) / batch_size)), elapsed=$t")
        end
    end
    return flow
end


# TODO: reconstruction makes a little sense for normalizing flows
# plot distribution in Z instead?
function reconstruct(flow::RealNVP, x::AbstractMatrix)
    z, _ = fwd_map(flow, x)
    x_rec = inv_map(flow, z)
    return x_rec
end


function show_pic(x)
    reshape(x, 28, 28)' |> imshow
end


function show_recon(m, x)
    x_ = reconstruct(m, x)
    show_pic(x)
    show_pic(x_)
end
