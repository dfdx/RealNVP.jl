
function xavier_init(T, dim_in, dim_out; c=1)
    low = -c * sqrt(6.0 / (dim_in + dim_out))
    high = c * sqrt(6.0 / (dim_in + dim_out))
    return T.(rand(Uniform(low, high), dim_in, dim_out))
end


function make_moons(;n_samples=100)
    n_samples_out = div(n_samples, 2)
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = cos.(LinRange(0, pi, n_samples_out))
    outer_circ_y = sin.(LinRange(0, pi, n_samples_out))
    inner_circ_x = 1 .- cos.(LinRange(0, pi, n_samples_out))
    inner_circ_y = 1 .- sin.(LinRange(0, pi, n_samples_out)) .- 0.5

    X = [outer_circ_x outer_circ_y; inner_circ_x  inner_circ_y]'
    X .+= 0.1 * randn(size(X))
    y = [zeros(Int, n_samples_out); ones(Int, n_samples_in)]
    return X, y
end
