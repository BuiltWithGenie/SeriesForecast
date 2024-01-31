module Delhi
using DataFrames, CSV, Dates, Statistics


features = [:meantemp, :humidity, :wind_speed, :meanpressure]
units = ["Celsius", "g/m³ of water", "km/h", "hPa"]
feature_names = ["Mean temperature", "Humidity", "Wind speed", "Mean pressure"]

"""Loads the entire Delhi dataset into a single dataframe."""
function load()
    df = vcat(
        CSV.read(pwd() * "/data/DailyDelhiClimateTrain.csv", DataFrame),
        CSV.read(pwd() * "/data/DailyDelhiClimateTest.csv", DataFrame),
    )
end

function normalize(x)
    μ = mean(x; dims=2)
    σ = std(x; dims=2)
    z = (x .- μ) ./ σ
    return z, μ, σ
end

function preprocess(raw_df, num_train=20)
    # Convert the date column into separate year and month columns as Float64
    raw_df[:, :year] = Float64.(year.(raw_df[:, :date]))
    raw_df[:, :month] = Float64.(month.(raw_df[:, :date]))

    # Aggregate the data by year and month, and compute the mean for the specified columns
    df = combine(
        groupby(raw_df, [:year, :month]),
        :date => (d -> mean(year.(d)) .+ mean(month.(d)) ./ 12), # Calculate average date
        :meantemp => mean,    # Average temperature
        :humidity => mean,    # Average humidity
        :wind_speed => mean,  # Average wind speed
        :meanpressure => mean,# Average pressure
        renamecols=false
    )
    df = select(df, Not([:year, :month]))

    # Split data into training and test sets based on the num_train parameter
    train_df = df[1:num_train, :]
    test_df = df[num_train + 1:end, :]

    # Normalize training data and get mean and var for both time and features
    t_train, t_mean, t_var = normalize(vec(train_df[!, :date])')
    y_train, y_mean, y_var = normalize(Matrix(train_df[!, Not(:date)])')

    # Normalize test data using training mean and var
    t_test = (vec(test_df[!, :date])' .- t_mean) ./ t_var
    y_test = (Matrix(test_df[!, Not(:date)])' .- y_mean) ./ y_var

    # Create DataFrames from the normalized arrays
    train_df = DataFrame(t = vec(t_train))
    test_df = DataFrame(t = vec(t_test))

    for (i, feature) in enumerate(features)
        train_df[!, feature] = y_train[i, :]
        test_df[!, feature] = y_test[i, :]
    end

    # Return DataFrames and normalization parameters
    return train_df, test_df, (t_mean = t_mean[1], t_var = t_var[1], y_mean = y_mean, y_var = y_var)
end

end
