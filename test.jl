include("lib/NodeUtils.jl")
include("lib/delhi.jl")
include("lib/utils.jl")
using .NodeUtils
using .Delhi
using JLD2, Statistics, DataFrames
using Interpolations, Observables
using Random


rng = MersenneTwister(123)
if isfile("data.jld2")
    @load "data.jld2" train_df test_df scaling
end

features = [:meantemp, :humidity, :wind_speed, :meanpressure]
units = ["Celsius", "g/m³ of water", "km/h", "hPa"]
feature_names = ["Mean temperature", "Humidity", "Wind speed", "Mean pressure"]


data = vcat(train_df, test_df)
# Functions to interpolate when calculating the MSE
interpolators = [LinearInterpolation(data.t, data[!, col]) for col in names(data)]
# NODE parameters
const obs_grid = 4:4:20 # we train on an increasing amount of the first k obs
const maxiters = 150
const lr = 5e-3
const N_steps = 100 # number of points in prediction over the full time range
_, θ_new, init_state = NodeUtils.neural_ode(train_df.t, length(features))
t_grid = range(minimum(data.t), maximum(data.t), length=N_steps) |> collect
θs = Observable(θ_new)
θs[], state = NodeUtils.train(Vector(train_df[!,:t]), Matrix(train_df[!,features]), obs_grid, lr, rng, θs; maxiters=maxiters);
