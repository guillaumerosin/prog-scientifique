using Dates
using CSV, DataFrames
using Statistics
using StateSpaceModels
using Plots

# ----------------------------------------------------------------------
# Analyse de séries temporelles sur le nombre moyen de vues "redhead"
# ----------------------------------------------------------------------

redhead = CSV.File("redhead.csv") |> DataFrame
sort!(redhead, :date)
select!(redhead, [:date, :nb_views])

y = Vector{Union{Missing, Float64}}(float.(redhead.nb_views))

function linear_interpolate!(v::Vector{Union{Missing, Float64}})
	idx = findall(!ismissing, v)
	if isempty(idx)
		error("Toutes les valeurs sont manquantes, interpolation impossible.")
	end

	if first(idx) > 1
		v[1:first(idx)-1] .= v[first(idx)]
	end

	if last(idx) < length(v)
		v[last(idx)+1:end] .= v[last(idx)]
	end

	for (i1, i2) in zip(idx[1:end-1], idx[2:end])
		y1, y2 = v[i1], v[i2]
		gap = i2 - i1
		if gap > 1
			step = (y2 - y1) / gap
			for k in 1:(gap-1)
				v[i1 + k] = y1 + step * k
			end
		end
	end

	return v
end

y_filled = linear_interpolate!(copy(y))
redhead.nb_views = collect(skipmissing(y_filled))  # plus de valeurs manquantes

λ = 0.5
boxcox(y::AbstractVector{<:Real}, λ) = (y .^ λ .- 1) ./ λ
inv_boxcox(z::AbstractVector{<:Real}, λ) = (λ .* z .+ 1) .^ (1 / λ)

y_bc = boxcox(redhead.nb_views, λ)

n = length(y_bc)
h = 30  # horizon de test (30 derniers jours)

if n <= h + 2
	error("Pas assez de données pour garder 30 jours pour le test.")
end

y_train = y_bc[1:n-h]
y_test_original = redhead.nb_views[n-h+1:end]

model1 = SARIMA(y_train; order = (1, 0, 1))
fit!(model1)

fc1 = forecast(model1, h)
yhat_bc = reduce(vcat, fc1.expected_value)
yhat = inv_boxcox(yhat_bc, λ)

mape = mean(abs.((y_test_original .- yhat) ./ y_test_original)) * 100
println("MAPE sur les 30 derniers jours = ", mape, " %")

model2 = SARIMA(y_bc; order = (1, 0, 1))
fit!(model2)

h_forecast = 120
fc2 = forecast(model2, h_forecast)
yfc_bc = reduce(vcat, fc2.expected_value)
yfc = inv_boxcox(yfc_bc, λ)

dates = redhead.date
dates_fc = collect(dates[end] + Day(1):Day(1):dates[end] + Day(h_forecast))

p = plot(dates, redhead.nb_views; label = "Données observées", xlabel = "Date",
         ylabel = "Nombre moyen de vues", legend = :topleft)
plot!(p, dates_fc, yfc; label = "Prévision 120 jours")

savefig(p, "redhead_forecast.png")
display(p)

println("Graphique affiché. Appuie sur Entrée pour fermer.")
readline()
