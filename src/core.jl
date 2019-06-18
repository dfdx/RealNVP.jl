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


include("utils.jl")
include("primitives.jl")
include("mlp.jl")
include("coupling.jl")
include("model.jl")
