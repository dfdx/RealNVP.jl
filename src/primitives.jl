relu(x::AbstractArray) = max.(x, 0)
relu_grad(x::AbstractArray) = float(x .> 0)  # TODO: ensure  relu_grad works for CuArray

@primitive relu(x::AbstractArray)
@primitive relu_grad(x::AbstractArray)

@grad relu(x::AbstractArray) 1 relu_grad(x)
