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

export neural_ode, train

include(joinpath( "delhi.jl"))
#= include(joinpath("..", "src", "figures.jl")) =#

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

function train_one_round(node, θ, state, y, opt, maxiters, rng, y0=y[:, 1]; kwargs...)
    predict(θ) = Array(node(y0, θ, state)[1])
    loss(θ) = sum(abs2, predict(θ) .- y)
    
    adtype = Optimization.AutoZygote()
    optf = OptimizationFunction((θ, p) -> loss(θ), adtype)
    optprob = OptimizationProblem(optf, θ)
    res = solve(optprob, opt, maxiters=maxiters; kwargs...)
    res.minimizer, state
end

function train(t, y, obs_grid, maxiters, lr, rng, θs , losses; kwargs...)
    log_results(θs, losses) =
        (θ, loss) -> begin
        push!(θs, copy(θ))
        push!(losses, loss)
        notify(parcopy)
        false
    end

    parcopy = θs
    #= θs, losses = ComponentArray[], Float32[] =#
    θ=nothing
    state=nothing
    
    for k in obs_grid
        @show k
        node, θ_new, state_new = neural_ode(t, size(y, 1))
        if θ === nothing θ = θ_new end
        if state === nothing state = state_new end

        θ, state = train_one_round(
            node, θ, state, y, Optimisers.ADAMW(lr), maxiters, rng;
            callback=log_results(θs[], losses[]),
            kwargs...
        )
        notify(θs)
    end
    θs, state, losses
end
end
