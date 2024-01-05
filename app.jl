using GenieFramework, PlotlyBase, JLD2, Statistics
using Interpolations
using Random
include("NodeUtils.jl")
using .NodeUtils
@genietools

disable_train = (haskey(ENV, "GENIE_ENV") && ENV["GENIE_ENV"] == "prod") ? "true" : "false"
button_color = disable_train == "true" ? "grey" : "primary"
button_tooltip = disable_train == "true" ? "Run the app locally to enable this button" : ""
#= include("node_utils.jl") =#
rng = MersenneTwister(123)
#include("delhi.jl")
#= df = Delhi.load() =#
#==#
#= t_train, y_train, t_test, y_test, (t_mean, t_scale), (y_mean, y_scale) = Delhi.preprocess(df) =#
#= @save "data.jld2" t_train y_train t_test y_test t_mean t_scale y_mean y_scale =#

@load "data.jld2" t_train y_train t_test y_test t_mean t_scale y_mean y_scale
@load "params.jld" θ

y = hcat(y_train, y_test)
t = vcat(t_train, t_test)
# Function to interpolate when calculating the MSE
interpolators = [LinearInterpolation(t, y[i,:]) for i in 1:4]
#= interp_func = LinearInterpolation(t, y) =#

calc_mse(t_predict, ŷ, interpolator) = round(mean((( ŷ - interpolator.(t_predict)).^2)), digits=3)

rescale_t(x) = t_scale .* x .+ t_mean
rescale_y(x,i) = y_scale[i] .* x .+ y_mean[i]

const obs_grid = 4:4:length(t_train) # we train on an increasing amount of the first k obs
const maxiters = 150
const lr = 5e-3
const N_steps = 100 # number of points in prediction over the full time range
#= θs, state, losses = train(t_train, y_train, obs_grid, maxiters, lr, rng, progress=true); =#
#= @save "training_output.bson" θs losses =#


function plot_pred(t, y, t̂, ŷ; kwargs...)
    traces = []
    plot_params = zip(eachrow(y), eachrow(ŷ), Delhi.feature_names, Delhi.units)
    for (i, (yᵢ, ŷᵢ, name, unit)) in enumerate(plot_params)
        trace_pred = scatter(x=t̂, y=ŷᵢ, mode="lines", name="Prediction", line=attr(color=i, width=3))
        trace_obs = scatter(x=t, y=yᵢ, mode="markers", name="Observation", marker=attr(size=5, color=i))
        push!(traces, trace_pred)
        push!(traces, trace_obs)
    end
    return traces
end

predict(y0, t, θ, state) = begin
    node, _, _ = neural_ode(t, length(y0))
    ŷ = Array(node(y0, θ, state)[1])
end

_, _, init_state = neural_ode(t_train, size(y_train, 1))
t_grid = collect(range(minimum(t_train), maximum(t_test), length=N_steps))
const ŷ_cached = predict(y_train[:,1], t_grid, θ, init_state)


get_layout(title, xlabel, ylabel) = PlotlyBase.Layout(
                                                      #= title=title, =#
                                                      xaxis=attr( title=xlabel, showgrid=false),
                                                      yaxis=attr( title=ylabel, showgrid=true),
                                                      margin=attr(l=5, r=5, t=5, b=5)
                                                     )
function get_traces(t_train, t_predict, y_train, ŷ, y_test, quantity_idx)
    [           
     PlotlyBase.scatter(x=rescale_t(t_predict), y=rescale_y(ŷ,quantity_idx), mode="line", name="ŷ"),
     PlotlyBase.scatter(x=rescale_t(t_train), y=rescale_y(y_train,quantity_idx), mode="markers", marker=attr(size=10, line=attr(width=2, color="DarkSlateGrey")), name = "y_train"),
                        PlotlyBase.scatter(x=rescale_t(t_test), y=rescale_y(y_test,quantity_idx), mode="markers", name = "y_test")
    ]
end

@app begin
    @in start=false
    @in animate=false
    @out disable_train = disable_train
    @out θs=[θ]
    @out losses = Float32[]
    @out temp_pdata = [PlotlyBase.scatter(x=[1,2,3])]
    @out hum_pdata = [PlotlyBase.scatter(x=[1,2,3])]
    @out wind_pdata = [PlotlyBase.scatter(x=[1,2,3])]
    @out press_pdata = [PlotlyBase.scatter(x=[1,2,3])]
    @out temp_layout = get_layout("Temperature", "Time", "Temperature")
    @out hum_layout = get_layout("Humidity", "Time", "Humidity")
    @out wind_layout = get_layout("Wind speed", "Time", "Wind speed")
    @out press_layout = get_layout("Mean pressure", "Time", "Mean pressure")
    @in predict=false
    @in r = 30
    @in pstep=1
    @out mse = [0.0,0.0,0.0,0.0]

    @private state = init_state
    @private k = 4
    @onbutton start begin
        println("Training")
        k=1
        _, state, _ = train(t_train, y_train, obs_grid, maxiters, lr, rng, __model__.θs, __model__.losses; progress=true);
        θ = θs[end]
        @save "params.jld" θ
        #= @save "training_output.bson" θs losses =#
    end
    @onchange r begin
        k = 4
        notify(__model__.θs)
    end
    @onchange isready, θs begin
        r_idx = 1:pstep:r
        #= t_grid = collect(range(minimum(t_train), maximum(t_test), length=Int(round(N_steps/pstep)))) =#
t_predict = t_grid[r_idx]
t_train_resc = t_train
#=         t_predict = rescale_t(t_grid[r_idx])[:] =#
#= t_train_resc = rescale_t(t_train)[:] =#
        #= ŷ = predict(y_train[:,1], t_predict, θs[end], state) =#
        ŷ = disable_train=="true" ? predict(y_train[:,1], t_predict, θs[end], state) : ŷ_cached[:,r_idx]
        temp_pdata = get_traces(t_train_resc, t_predict, y_train[1,1:obs_grid[k]], ŷ[1,r_idx], y_test[1,:], 1)
        hum_pdata = get_traces(t_train_resc, t_predict, y_train[2,1:obs_grid[k]], ŷ[2,r_idx], y_test[2,:], 2)
        wind_pdata = get_traces(t_train_resc, t_predict, y_train[3,1:obs_grid[k]], ŷ[3,r_idx], y_test[3,:], 3)
        press_pdata = get_traces(t_train_resc, t_predict, y_train[4,1:obs_grid[k]], ŷ[4,r_idx], y_test[4,:], 4)
        # Now you can interpolate to find y at new x values
        #= mse = mean((( ŷ[:,r_idx] .- interp_func.(t_predict)).^2),dims=2)[:] =#
        mse = [calc_mse(t_predict, ŷ[i,:], interpolators[i]) for i in 1:4]
        #= mse = mean((( ŷ[:,r_idx] .- y[:,r_idx]).^2),dims=2)[:] =#
        k += 1
    end

    @onbutton predict begin
        ŷ = predict(y_train[:,1], t_grid, θs[end], state)
    end
    @onbutton animate begin
        for i in 30:pstep:100
            r = i
            sleep(0.1)
        end
    end
end

ui() =[
       h1("train and predict"),btn("Train", @click(:start), loading=:start), 
       range(1:100,:r),
       cell(class="row"),
       GenieFramework.plot(:temp_pdata, layout=:temp_layout), 
       GenieFramework.plot(:hum_pdata, layout=:hum_layout),
       GenieFramework.plot(:wind_pdata, layout=:wind_layout),
       GenieFramework.plot(:press_pdata, layout=:press_layout)
      ]
@page("/","app.jl.html")
