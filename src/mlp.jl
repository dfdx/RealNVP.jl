mutable struct MLP3{T}
    W1::AbstractMatrix{T}
    b1::AbstractVector{T}
    W2::AbstractMatrix{T}
    b2::AbstractVector{T}
end


MLP3{T}(i, h, o) where T = MLP3(xavier_init(T, h, i), zeros(T, h),
                                xavier_init(T, o, h), zeros(T, o))

MLP3(i, h, o) = MLP3{Float64}(i, h, o)


function Base.show(io::IO, m::MLP3{T}) where T
    h, i = size(m.W1)
    o = size(m.W2, 1)
    print(io, "MLP{$T}($i, $h, $o)")
end


function (m::MLP3{T})(x) where T
    x = relu(m.W1 * x .+ m.b1)
    x = relu(m.W2 * x .+ m.b2)
    return x
end


# MLP with tanh() activation
# TODO: it's quite stupid design, but I'll fix it later

mutable struct MLP3Tanh{T}
    W1::AbstractMatrix{T}
    b1::AbstractVector{T}
    W2::AbstractMatrix{T}
    b2::AbstractVector{T}
end


MLP3Tanh{T}(i, h, o) where T = MLP3Tanh(xavier_init(T, h, i), zeros(T, h),
                                        xavier_init(T, o, h), zeros(T, o))

MLP3Tanh(i, h, o) = MLP3Tanh{Float64}(i, h, o)


function Base.show(io::IO, m::MLP3Tanh{T}) where T
    h, i = size(m.W1)
    o = size(m.W2, 1)
    print(io, "MLPTanh{$T}($i, $h, $o)")
end


function (m::MLP3Tanh{T})(x) where T
    x = relu(m.W1 * x .+ m.b1)
    x = relu(m.W2 * x .+ m.b2)
    tanh.(x)
    return x
end



#############################################

mlp_loss(m, x, y) = sum(abs2.(y .- m(x)))


function main_mlp()
    # verified - work fine!
    lr = 1e-5
    X, y = make_moons()
    x = X[:, 1:10]
    m = MLP3Tanh(2, 256, 2)
    for epoch=1:1000
        epoch_cost = 0
        for (i, x) in enumerate(eachbatch(X, size=10))
            cost, g = grad(mlp_loss, m, x, 2 .* x)
            update!(m, g[1], (x, gx) -> x .- lr * gx)
            epoch_cost += cost
        end
        println(epoch_cost)
    end
end


function cross_check_mlp()
    X, y = make_moons()
    x = X[:, 1:10]
    m = MLP3Tanh(2, 256, 2)
    grad(mlp_loss, m, x, 2 .* x)[2][2]
    Zygote.gradient(mlp_loss, m, x, 2 .* x)[2]
end
