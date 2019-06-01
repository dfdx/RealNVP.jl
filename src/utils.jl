
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


# def make_moons(n_samples=100, shuffle=True, noise=None, random_state=None):
#     """Make two interleaving half circles
#     A simple toy dataset to visualize clustering and classification
#     algorithms. Read more in the :ref:`User Guide <sample_generators>`.
#     Parameters
#     ----------
#     n_samples : int, optional (default=100)
#         The total number of points generated.
#     shuffle : bool, optional (default=True)
#         Whether to shuffle the samples.
#     noise : double or None (default=None)
#         Standard deviation of Gaussian noise added to the data.
#     random_state : int, RandomState instance or None (default)
#         Determines random number generation for dataset shuffling and noise.
#         Pass an int for reproducible output across multiple function calls.
#         See :term:`Glossary <random_state>`.
#     Returns
#     -------
#     X : array of shape [n_samples, 2]
#         The generated samples.
#     y : array of shape [n_samples]
#         The integer labels (0 or 1) for class membership of each sample.
#     """

#     n_samples_out = n_samples // 2
#     n_samples_in = n_samples - n_samples_out

#     generator = check_random_state(random_state)

#     outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
#     outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
#     inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
#     inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - .5

#     X = np.vstack([np.append(outer_circ_x, inner_circ_x),
#                    np.append(outer_circ_y, inner_circ_y)]).T
#     y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
#                    np.ones(n_samples_in, dtype=np.intp)])

#     if shuffle:
#         X, y = util_shuffle(X, y, random_state=generator)

#     if noise is not None:
#         X += generator.normal(scale=noise, size=X.shape)

#     return X, y
