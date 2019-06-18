relu(x::AbstractArray) = max.(zero(eltype(x)), x)
relu_grad(ds::Real, y::Real) = ifelse(y > 0, ds, zero(typeof(y)))
relu_grad(ds::AbstractArray, y::AbstractArray) = relu_grad.(ds, y)

@diffrule relu(x::AbstractArray) x relu_grad(ds, relu(x))

leaky_relu(x::T, alpha) where T <: Real = max(T(alpha) * x, x)
leaky_relu(x::AbstractArray, alpha) = leaky_relu.(x, alpha)

leaky_relu_grad(ds::Real, y::T, alpha) where T <: Real = ifelse(y > 0, ds, T(alpha))
leaky_relu_grad(x::AbstractArray, alpha) = leaky_relu_grad.(x, alpha)

@diffrule leaky_relu(x::AbstractArray, _alpha::Real) x leaky_relu_grad(x, _alpha)
@diffrule leaky_relu(x::AbstractArray, _alpha::Real) _alpha 0
