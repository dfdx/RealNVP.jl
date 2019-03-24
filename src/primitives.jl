relu(x::AbstractArray) = max.(x, 0)
relu_grad(x::AbstractArray) = float(x .> 0)  # TODO: ensure  relu_grad works for CuArray

@diffrule relu(x::AbstractArray) x relu_grad(x)
