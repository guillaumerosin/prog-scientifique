using Pkg
Pkg.activate("..")
using CSV, DataFrames, Statistics, Plots

files = readdir("data", join=true)
dfs = Dict(basename(f) => CSV.read(f, DataFrame) for f in files)

for (name, df) in dfs
    println("\n $name ")
    println("Colonnes : ", names(df))
    println("Dimensions : ", size(df))
end

# Je vais isoler One Piece dans all-series-ep-average.csv
my_series = dfs["all-series-ep-average.csv"]
one_piece = filter(row -> occursin("One Piece", row.Title), my_series)
println(one_piece)

# Je vais isoler One Piece dans all-episode-ratings.csv
op_code = one_piece.Code[1]
all_eps = dfs["all-episode-ratings.csv"]
op_eps = filter(row -> row.Code == op_code, all_eps)
println("\n One Piece episodes :")
println(op_eps)

# Mtn j'explore les données d'One Piece groupe d'épisodes
# === Récupérer les épisodes One Piece ===
ep_ratings = dfs["all-episode-ratings.csv"]
op_ep = filter(row -> row.Code == "tt0388629", ep_ratings)
sort!(op_ep, :Episode)

println("Nombre d'épisodes : ", nrow(op_ep))

# === Découper en arcs de ~50 épisodes ===
arc_size = 50
op_ep[!, :Arc] = ceil.(Int, op_ep[!, :Episode] ./ arc_size)

# === Moyenne par arc ===
arcs = combine(groupby(op_ep, :Arc),
    :Rating => mean => :Rating_Mean,
    :Rating => length => :Nb_Episodes
)
sort!(arcs, :Arc)
println("\n=== Ratings par arc (groupes de $arc_size épisodes) ===")
println(arcs)

# === Régression linéaire sur les arcs ===
n = nrow(arcs)
x = collect(1:n)
y = arcs[!, :Rating_Mean]

a = (n * sum(x .* y) - sum(x) * sum(y)) / (n * sum(x .^ 2) - sum(x)^2)
b = (sum(y) - a * sum(x)) / n

println("\nTendance (pente) : ", round(a, digits=4))
next_arc = n + 1
prediction = a * next_arc + b
println("Prédiction arc suivant : ", round(prediction, digits=2))
