module NodeUtils
using Random
using Dates
using Optimization
using Lux
using OptimizationOptimisers
using DiffEqFlux: NeuralODE, swish
using DifferentialEquations
using ComponentArrays
using Observables

export neural_ode, train, predict

include(joinpath( "delhi.jl"))
#= include(joinpath("..", "src", "figures.jl")) =#

"""
    neural_ode(t, data_dim)

Create a neural ordinary differential equation (Neural ODE) model.
It initializes the model with a neural network, which maps the input data dimension
to the output data dimension through hidden layers. Returns the
neural ODE model, its parameters, and state.
"""
function neural_ode(t, data_dim)
    f = Lux.Chain(
        Lux.Dense(data_dim, 64, swish),
        Lux.Dense(64, 32, swish),
        Lux.Dense(32, data_dim)
    )

    node = NeuralODE(
        f, extrema(t), Tsit5(),
        saveat=t,
        abstol=1e-9, reltol=1e-9
    )
    
    rng = Random.default_rng()
    p, state = Lux.setup(rng, f)

    return node, ComponentArray(p), state
end

"""
    train_one_round(node, θ, state, y, opt, maxiters, rng, y0=y[1, :]; kwargs...)

Perform one round of training for the neural ODE model.
Updates the model's parameters by minimizing the loss function
using the specified optimizer. Returns the updated parameters and state.
"""
function train_one_round(node, θ, state, y, opt, maxiters, rng, y0=y[1, :]; kwargs...)
    @show "innn"
    predict(θ) = Array(node(y0, θ, state)[1])
    loss(θ) = sum(abs2, predict(θ)' .- y)
    @show size(y), size(y0), size(predict(θ))
    
    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction((θ, p) -> loss(θ), adtype)
    optprob = OptimizationProblem(optf, θ)
    res = solve(optprob, opt, maxiters=maxiters; kwargs...)
    res.minimizer, state
end

"""
    train(t, y, obs_grid, maxiters, lr, rng, θs, losses; kwargs...)

Train the neural ODE model over a grid of observations.
Iteratively updates the model parameters and logs the results.
Returns the collection of parameter updates, final state, and losses.
θs is an Observable that is udpdated after every training step
"""
function train(t, y, obs_grid, maxiters, lr, rng, θs; kwargs...)
    θ=nothing
    state=nothing
    
    for k in obs_grid
        node, θ_new, state_new = neural_ode(t[1:k], size(y, 2))
        if θ === nothing θ = θ_new end
        if state === nothing state = state_new end

        θ, state = train_one_round( node, θ, state, y[1:k,:], Optimisers.ADAMW(lr), maxiters, rng; kwargs...)
        @show k, size(y)
        θs[] = θ
    end
    θs[], state
end


predict(y0, t, θ, state) = begin
    node, _, _ = neural_ode(t, length(y0))
    ŷ = Array(node(y0, θ, state)[1])
end
end

function mod(x)
    @show x
    x = x+1
end
