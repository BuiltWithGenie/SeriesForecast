using GenieFramework, PlotlyBase, JLD2, Statistics, DataFrames
using Interpolations
using Random
#= include("NodeUtils.jl") =#
using .NodeUtils
using .Delhi
#= include("delhi.jl") =#
#= include("utils.jl") =#
@genietools

prod_mode = (haskey(ENV, "GENIE_ENV") && ENV["GENIE_ENV"] == "prod") ? "true" : "false"
button_color = prod_mode == "true" ? "grey" : "primary"
button_tooltip = prod_mode == "true" ? "Run the app locally to enable this button" : ""

rng = MersenneTwister(123)
if isfile("data.jld2")
    @load "data.jld2" train_df test_df scaling
else
    rawdata = Delhi.load()
    train_df, test_df, scaling = Delhi.preprocess(rawdata)
    @save "data.jld2" train_df test_df scaling
end

if isfile("params.jld")
    @load "params.jld" θ
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
_, θ_new, init_state = neural_ode(train_df.t, length(features))
t_grid = range(minimum(data.t), maximum(data.t), length=N_steps) |> collect
# We cache the predictions for the full time range to avoid recomputing them
#= const ŷ_cached = predict(Vector(train_df[1,features]), t_grid, θ, init_state) =#

@app begin
    @in start=false
    @in animate=false
    @out prod_mode = prod_mode
    @out θ=θ_new
    @out losses = Float32[]
    @out temp_pdata = [PlotlyBase.scatter(x=[1,2,3])]
    @out hum_pdata = [PlotlyBase.scatter(x=[1,2,3])]
    @out wind_pdata = [PlotlyBase.scatter(x=[1,2,3])]
    @out press_pdata = [PlotlyBase.scatter(x=[1,2,3])]
    @out temp_layout = get_layout("Temperature", "Time", "Temperature")
    @out hum_layout = get_layout("Humidity", "Time", "Humidity")
    @out wind_layout = get_layout("Wind speed", "Time", "Wind speed")
    @out press_layout = get_layout("Mean pressure", "Time", "Mean pressure")
    @in r = 30
    @in pstep=1
    @out mse = [0.0,0.0,0.0,0.0]
    @in k = 20

    @private state = init_state
    @onbutton start begin
        println("Training")
        # change k and r to display the correct number of training points and the prediction over the entire range
        k=1; r[!] = 100;
        @show size(Matrix(train_df[!,features]))
        # We pass the Observable version of θ to `train` , which will update its value during training.
        # When the training is finished we store the final value in θ
        θ, state = train(Vector(train_df[!,:t]), Matrix(train_df[!,features]), obs_grid, maxiters, lr, rng, __model__.θ; progress=true);
        @save "params.jld" θ
    end
    @onchange r begin
        k = 20
    end
    # when θ is upgraded during a training loop, increase the number of k training points
    # shown in the plots
    @onchange θ begin
        k += 4
    end
    @onchange isready, θ, r begin
        r_idx = 1:pstep:r
        t_predict = t_grid[r_idx]
        #= ŷ = prod_mode=="true" ? predict(y_train[:,1], t_predict, θs[end], state) : ŷ_cached[:,r_idx] =#
        ŷ = predict(Vector(train_df[1,features]), t_predict, θ, state)
        predict_df = DataFrame(t = t_predict, meantemp = ŷ[1,:], humidity = ŷ[2,:], wind_speed = ŷ[3,:], meanpressure = ŷ[4,:])
        temp_pdata, hum_pdata, wind_pdata, press_pdata = get_traces(data[1:k,:], data[k+1:end,:], predict_df, scaling)

        mse = [calc_mse(t_predict, ŷ[i,:], interpolators[i]) for i in 1:4]
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
