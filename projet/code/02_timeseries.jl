# 02_timeseries.jl — Construction de la série temporelle One Piece par saga
# Ch3 : modélisation d'une série temporelle, traitement des missing
# Ch4 : intégration numérique par la méthode des trapèzes

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using CSV, DataFrames, Statistics, Plots, Printf
import Random
Random.seed!(42)

DATA_DIR      = joinpath(@__DIR__, "..", "data")
RESULTS_DIR   = joinpath(@__DIR__, "..", "results")
isdir(RESULTS_DIR) || mkdir(RESULTS_DIR)

OP_CODE = "tt0388629"

# ── 1. Chargement et filtrage One Piece ───────────────────────────────────────
ep_df = CSV.read(joinpath(DATA_DIR, "all-episode-ratings.csv"), DataFrame;
                 missingstring=["NA","","N/A"])

# filter : garde uniquement les lignes où Code == OP_CODE
op = filter(row -> !ismissing(row.Code) && row.Code == OP_CODE, ep_df)
sort!(op, :Episode)   # sécurité : s'assurer que les épisodes sont dans l'ordre

@printf("Épisodes One Piece chargés : %d  (eps. 1 à %d)\n", nrow(op), maximum(op.Episode))

# ── 2. Définition des sagas officielles ───────────────────────────────────────
# Source : One Piece Wiki — découpage par saga anime
# Les 6 derniers épisodes (878-883) appartiennent à l'arc Reverie ;
# trop peu pour une analyse seule, on les fusionne avec Zou + WCI
# en une "Saga des Quatre Empereurs".
const SAGAS = [
    (1,  "East Blue",              1,   61),
    (2,  "Alabasta",              62,  135),
    (3,  "Sky Island",           136,  206),
    (4,  "Water 7",              207,  325),
    (5,  "Thriller Bark",        326,  384),
    (6,  "Summit War",           385,  516),
    (7,  "Fish-Man Island",      517,  574),
    (8,  "Dressrosa",            575,  746),
    (9,  "Quatre Empereurs",     747,  883),   # Zou + WCI + Reverie
]

# ── 3. Assignation des étiquettes de saga ─────────────────────────────────────
# On ajoute une colonne SagaId et SagaName à chaque épisode
op[!, :SagaId]   = zeros(Int, nrow(op))
op[!, :SagaName] = fill("", nrow(op))

for (id, name, ep_start, ep_end) in SAGAS
    mask = (op.Episode .>= ep_start) .& (op.Episode .<= ep_end)
    op[mask, :SagaId]   .= id
    op[mask, :SagaName] .= name
end

# Vérification : épisodes non assignés ?
unassigned = count(op.SagaId .== 0)
@printf("Épisodes non assignés : %d\n", unassigned)

# ── 4. Intégration numérique par trapèzes (Ch4) ───────────────────────────────
#
# Pour chaque saga on a une suite de notes r₁, r₂, …, rₙ (une par épisode).
# On les traite comme des valeurs d'une fonction r(t) aux points t = 1,2,…,n.
#
# Règle des trapèzes sur n points équidistants (Δt = 1) :
#   ∫r(t)dt ≈ Δt · [r₁/2 + r₂ + r₃ + … + rₙ₋₁ + rₙ/2]
#
# Note pondérée de la saga = intégrale / (n - 1)   [longueur de l'intervalle]
#
# Différence par rapport à la moyenne simple :
#   - La moyenne simple donne un poids égal à chaque épisode.
#   - La règle des trapèzes donne un demi-poids au 1er et au dernier épisode.
#   - Pour les grandes sagas l'écart est minime ; pour les petites (~10 éps)
#     il peut atteindre quelques centièmes de point.
# C'est pertinent quand une saga démarre fort puis finit abruptement (ou inverse).
#
function trapeze_mean(ratings::AbstractVector{<:Real})
    n = length(ratings)
    n == 1 && return Float64(ratings[1])   # cas dégénéré : un seul épisode
    # Δt = 1 → intégrale ≈ somme des trapèzes successifs
    integral = ratings[1]/2 + sum(ratings[2:end-1]) + ratings[end]/2
    return integral / (n - 1)             # normalisation par la longueur
end

# ── 5. Calcul de la série temporelle ──────────────────────────────────────────
# combine : applique plusieurs fonctions sur chaque groupe (saga)
ts = combine(groupby(op, [:SagaId, :SagaName]),
    :Rating => mean                  => :MoySimple,    # moyenne arithmétique
    :Rating => trapeze_mean          => :MoyTrapeze,   # intégrale normalisée
    :Rating => std                   => :Std,
    :Rating => minimum               => :Min,
    :Rating => maximum               => :Max,
    :Rating => length                => :NbEps,
)
sort!(ts, :SagaId)

# ── 6. Affichage ──────────────────────────────────────────────────────────────
println("\n", "="^72)
println("SÉRIE TEMPORELLE ONE PIECE — note par saga")
println("="^72)
@printf("%-4s  %-22s  %6s  %9s  %9s  %6s  %5s-%5s\n",
        "Saga", "Nom", "NbEps", "MoySimple", "Trapèze", "Std", "Min", "Max")
println("-"^72)
for row in eachrow(ts)
    @printf("%-4d  %-22s  %6d  %9.4f  %9.4f  %6.4f  %5.1f  %5.1f\n",
            row.SagaId, row.SagaName, row.NbEps,
            row.MoySimple, row.MoyTrapeze, row.Std,
            row.Min, row.Max)
end

# ── 7. Différence trapèzes vs moyenne simple ──────────────────────────────────
println("\nÉcart max entre trapèze et moyenne simple : ",
        @sprintf("%.5f", maximum(abs.(ts.MoyTrapeze .- ts.MoySimple))),
        " point (confirme que la différence est faible mais non nulle)")

# ── 8. Sauvegarde du résultat ─────────────────────────────────────────────────
CSV.write(joinpath(RESULTS_DIR, "one_piece_sagas.csv"), ts)
println("\nFichier sauvegardé → results/one_piece_sagas.csv")

# ── 9. Visualisation ──────────────────────────────────────────────────────────
# Courbe des notes moyennes par saga avec barres d'erreur ± std
p = plot(
    ts.SagaId,
    ts.MoySimple;
    yerror       = ts.Std,
    marker       = :circle,
    markersize   = 6,
    linewidth    = 2,
    color        = :royalblue,
    label        = "Note moyenne (simple)",
    xlabel       = "Saga (ordre chronologique)",
    ylabel       = "Note IMDb moyenne",
    title        = "One Piece — Évolution de la note par saga",
    xticks       = (ts.SagaId, ts.SagaName),
    xrotation    = 30,
    legend       = :bottomleft,
    size         = (900, 500),
    bottom_margin = 20Plots.mm,
    grid         = true,
)

# Superposition de la note pondérée par trapèzes
plot!(p,
    ts.SagaId,
    ts.MoyTrapeze;
    marker    = :diamond,
    markersize = 5,
    linestyle = :dash,
    color     = :orange,
    label     = "Note pondérée (trapèzes)",
)

# Ligne de seuil "succès" à 8.0
hline!(p, [8.0];
    color     = :red,
    linestyle = :dot,
    linewidth = 1.5,
    label     = "Seuil succès (8.0)",
)

savefig(p, joinpath(RESULTS_DIR, "one_piece_sagas.png"))
println("Graphique sauvegardé → results/one_piece_sagas.png")

println("\n✔  Série temporelle construite (", nrow(ts), " points).")
