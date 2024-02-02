module Utils
using PlotlyBase, DataFrames, Statistics, Interpolations
export rescale_t, rescale_y, calc_mse, plot_pred, get_layout, get_traces

rescale_t(x) = t_scale .* x .+ t_mean
rescale_y(x,i) = y_scale[i] .* x .+ y_mean[i]

calc_mse(t_predict, ŷ, interpolator) = round(mean((( ŷ - interpolator.(t_predict)).^2)), digits=3)

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

get_layout(title, xlabel, ylabel) = PlotlyBase.Layout(
    #= title=title, =#
    xaxis=attr( title=xlabel, showgrid=false),
    yaxis=attr( title=ylabel, showgrid=true),
    #= margin=attr(l=5, r=5, t=15, b=5), =#
    legend=attr( x=1, y=1.02, yanchor="bottom", xanchor="right", orientation="h"),
   )

function get_traces(t_train, t_predict, y_train, ŷ, y_test, quantity_idx)
    [           
     PlotlyBase.scatter(x=rescale_t(t_predict), y=rescale_y(ŷ,quantity_idx), mode="line", name="ŷ",line=attr(color="black")),
     PlotlyBase.scatter(x=rescale_t(t_train), y=rescale_y(y_train,quantity_idx), mode="markers", marker=attr(size=10, line=attr(width=2, color="DarkSlateGrey")), name = "y_train"),
     PlotlyBase.scatter(x=rescale_t(t_test), y=rescale_y(y_test,quantity_idx), mode="markers", name = "y_test", marker=attr(size=6, color="orange"))
    ]
end

function get_traces(train_df, test_df, predict_df, norm)
    # Function to rescale time
    rescale_t(x) = norm.t_var .* x .+ norm.t_mean

    # Function to rescale a specific feature
    rescale_y(y, i) = norm.y_var[i] .* y .+ norm.y_mean[i]

    # Initialize an empty array to store lists of traces
    all_traces = []

    # Iterate over the columns, skipping the 't' column
    for (i, col_name) in zip(1:4, names(train_df)[2:end])
            # Generate traces for each feature
            feature_traces = [
                PlotlyBase.scatter(x=rescale_t(predict_df.t), y=rescale_y(predict_df[!, col_name], i), mode="line", line=attr(color="black"), name="Predict"),
                PlotlyBase.scatter(x=rescale_t(train_df.t), y=rescale_y(train_df[!, col_name], i), mode="markers", marker=attr(size=10, color="darkcyan", line=attr(width=0, color="DarkSlateGrey")), name = "Train"),
                PlotlyBase.scatter(x=rescale_t(test_df.t), y=rescale_y(test_df[!, col_name], i), mode="markers", marker=attr(size=6, color="orange"), name = "Test")
            ]

            # Add the set of traces to the list
            push!(all_traces, feature_traces)
    end

    return all_traces
end
end
