using GenieFramework,  PlotlyBase
using BSON: @save, @load
@genietools

include("node_utils.jl")
rng = MersenneTwister(123)
df = Delhi.load()
#= plt_features = Delhi.plot_features(df) =#
#= savefig(plt_features, joinpath("plots", "features.svg")) =#

#= df_2016 = filter(x -> x.date < Date(2016, 1, 1), df) =#
#= plt_2016 = plot( =#
#=     df_2016.date, =#
#=     df_2016.meanpressure, =#
#=     title = "Mean pressure, before 2016", =#
#=     ylabel = Delhi.units[4], =#
#=     xlabel = "Time", =#
#=     color = 4, =#
#=     size = (600, 300), =#
#=     label = nothing, =#
#=     right_margin=5Plots.mm =#
#= ) =#
#= savefig(plt_2016, joinpath("plots", "zoomed_pressure.svg")) =#

t_train, y_train, t_test, y_test, (t_mean, t_scale), (y_mean, y_scale) = Delhi.preprocess(df)
@save "data.jld2" t_train y_train t_test y_test t_mean t_scale y_mean y_scale

@load "data.jld2" t_train y_train t_test y_test t_mean t_scale y_mean y_scale
#= @load "params.jld" θ =#


rescale_t(x) = t_scale .* x .+ t_mean
rescale_y(x,i) = y_scale[i] .* x .+ y_mean[i]

obs_grid = 4:4:length(t_train) # we train on an increasing amount of the first k obs
maxiters = 100
lr = 5e-3
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
    println("aaaa")
    ŷ = Array(node(y0, θ, state)[1])
end

t_grid = collect(range(minimum(t_train), maximum(t_test), length=100))

node, θ_new, init_state = neural_ode(t_train, size(y_train, 1))
get_layout(title, xlabel, ylabel) = PlotlyBase.Layout(
                                                      title=title,
                                                      xaxis=attr(
                                                                 title=xlabel,
                                                                 showgrid=false
                                                                ),
                                                      yaxis=attr(
                                                                 title=ylabel,
                                                                 showgrid=true,
                                                                )
                                                     )
@app begin
    @in start=false
    @out θs=ComponentArray[]
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
    @in r = RangeData(1:100)

    @private state = init_state
    @private k = 1
    #= @private θ::Any = 3 =#
    @onbutton start begin
        println("Training")
        _, state, _ = train(t_train, y_train, obs_grid, maxiters, lr, rng, __model__.θs, __model__.losses; progress=true);
        θ = θs[end]
        @show "here"
        @save "params.jld" θ
        k=1
        #= @save "training_output.bson" θs losses =#
    end
    @onchange θs,r begin
        @show r
        r_idx = r.range.start:r.range.stop
        t_predict = t_grid[r_idx]
        ŷ = predict(y_train[:,1], t_predict, θs[end], state)
        temp_pdata = [
                      PlotlyBase.scatter(x=t_predict, y=rescale_y(ŷ[1,r_idx],1), mode="line"),
                      PlotlyBase.scatter(x=t_train, y=rescale_y(y_train[1,1:obs_grid[k]],1), mode="markers")
                     ]
        hum_pdata = [
                     PlotlyBase.scatter(x=t_predict, y=rescale_y(ŷ[2,r_idx],2), mode="line"),
                     PlotlyBase.scatter(x=t_train, y=rescale_y(y_train[2,1:obs_grid[k]],2), mode="markers")
                    ]
        wind_pdata = [
                      PlotlyBase.scatter(x=t_predict, y=rescale_y(ŷ[3,r_idx],3), mode="line"),
                      PlotlyBase.scatter(x=t_train, y=rescale_y(y_train[3,1:obs_grid[k]],3), mode="markers")
                     ]
        press_pdata = [
                       PlotlyBase.scatter(x=t_predict, y=rescale_y(ŷ[4,r_idx],4), mode="line"),
                       PlotlyBase.scatter(x=t_train, y=rescale_y(y_train[4,1:obs_grid[k]],4), mode="markers")
                      ]
        k += 1
    end
    @onbutton predict begin
        ŷ = predict(y_train[:,1], t_grid, θs[end], state)
    end
end

ui() =[
       h1("train and predict"),btn("Train", @click(:start), loading=:start), 
       range(1:100,:r),
       GenieFramework.plot(:temp_pdata, layout=:temp_layout), 
       GenieFramework.plot(:hum_pdata, layout=:hum_layout),
       GenieFramework.plot(:wind_pdata, layout=:wind_layout),
       GenieFramework.plot(:press_pdata, layout=:press_layout)
      ]
@page("/","app.jl.html")
