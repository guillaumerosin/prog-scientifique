# 01_explore.jl — Exploration initiale des 8 fichiers CSV du dataset IMDb
# Ch2 : projet reproductible (Pkg.activate + seed fixée)
# Ch3 : détection et comptage des valeurs manquantes

# ── Activation de l'environnement isolé (Ch2 : reproductibilité) ──────────────
import Pkg
Pkg.activate(@__DIR__)   # charge le Project.toml situé dans code/
Pkg.instantiate()        # installe les dépendances manquantes si nécessaire

using CSV, DataFrames, Statistics, Printf

# Seed globale (Ch2 : tout tirage aléatoire ultérieur donnera le même résultat)
import Random
Random.seed!(42)

# ── Chemins ───────────────────────────────────────────────────────────────────
DATA_DIR = joinpath(@__DIR__, "..", "data")

CSV_FILES = [
    "00.imdb_top_250_series_episode_ratings.csv",
    "00.imdb_top_250_series_global_ratings.csv",
    "all-episode-ratings.csv",
    "all-series-ep-average.csv",
    "top250list.csv",
    "top-250-movie-ratings.csv",
    "top-seasons-full.csv",
    "top-seasons-more-than-2-eps.csv",
]

# ── Taux de valeurs manquantes pour une colonne (Ch3) ────────────────────────
# `missing` est une valeur sentinelle en Julia (comme NA en R).
# `ismissing(x)` renvoie true uniquement pour cette valeur spéciale.
function missing_rate(col)
    n = length(col)
    n == 0 && return 0.0
    return 100.0 * count(ismissing, col) / n
end

# ── Petite fonction d'affichage d'un tableau texte ───────────────────────────
# Affiche un DataFrame sous forme de tableau aligné, sans dépendance externe
function print_table(df::DataFrame; max_rows=3, max_col_width=22)
    cols  = names(df)
    ncols = length(cols)

    # Largeur de chaque colonne = max(longueur nom, longueur valeur tronquée)
    widths = Int[]
    for c in cols
        w = length(c)
        for i in 1:min(max_rows, nrow(df))
            v = string(df[i, c])
            w = max(w, min(length(v), max_col_width))
        end
        push!(widths, w)
    end

    sep = "+" * join(("-"^(w+2) for w in widths), "+") * "+"

    println(sep)
    header = "| " * join(rpad(c, widths[i]) for (i,c) in enumerate(cols)) * " |"
    # corrige l'espacement entre colonnes
    header = "| " * join((rpad(cols[i], widths[i]) for i in 1:ncols), " | ") * " |"
    println(header)
    println(sep)
    for r in 1:min(max_rows, nrow(df))
        row_str = "| " * join((rpad(first(string(df[r,c]), max_col_width), widths[i])
                                for (i,c) in enumerate(cols)), " | ") * " |"
        println(row_str)
    end
    println(sep)
end

# ── Boucle principale d'exploration ───────────────────────────────────────────
for fname in CSV_FILES
    path = joinpath(DATA_DIR, fname)
    # missingstring : liste des chaînes à interpréter comme `missing` (Ch3)
    df = CSV.read(path, DataFrame; missingstring=["NA", "", "N/A"])

    println("\n", "="^70)
    println("FICHIER : ", fname)
    @printf("  Lignes : %d   |   Colonnes : %d\n", nrow(df), ncol(df))

    println("\n  Colonnes / Types / % manquant :")
    @printf("  %-25s  %-35s  %s\n", "Colonne", "Type Julia", "% manquant")
    println("  ", "-"^65)
    for c in names(df)
        rate = missing_rate(df[!, c])
        @printf("  %-25s  %-35s  %.1f %%\n",
                c,
                string(eltype(df[!, c])),
                rate)
    end

    println("\n  3 premières lignes :")
    print_table(df; max_rows=3)
end

# ── Présence de One Piece dans chaque fichier ─────────────────────────────────
OP_CODE  = "tt0388629"
OP_REGEX = r"one piece"i   # expression régulière insensible à la casse

println("\n\n", "="^70)
println("PRÉSENCE DE ONE PIECE (tt0388629) PAR FICHIER")
println("="^70)

for fname in CSV_FILES
    path = joinpath(DATA_DIR, fname)
    df   = CSV.read(path, DataFrame; missingstring=["NA", "", "N/A"])

    found = false
    for c in names(df)
        T = eltype(df[!, c])
        if T <: Union{Missing, AbstractString}
            if any(x -> !ismissing(x) && (x == OP_CODE || occursin(OP_REGEX, x)),
                   df[!, c])
                found = true
                break
            end
        end
    end
    status = found ? "TROUVÉ  ✓" : "absent  ✗"
    println("  $status  →  $fname")
end

# ── Zoom One Piece : épisodes ─────────────────────────────────────────────────
println("\n\n", "="^70)
println("DONNÉES ONE PIECE — all-episode-ratings.csv")
println("="^70)

ep_df  = CSV.read(joinpath(DATA_DIR, "all-episode-ratings.csv"), DataFrame;
                  missingstring=["NA", "", "N/A"])

# La colonne Code est en 5e position (index 1-based en Julia)
op_eps = filter(row -> !ismissing(row.Code) && row.Code == OP_CODE, ep_df)

@printf("  Épisodes trouvés : %d\n", nrow(op_eps))

# unique() retourne les valeurs distinctes ; sort() les trie
seasons = sort(unique(skipmissing(op_eps.Season)))
println("  Saisons présentes : ", collect(seasons))

println("\n  Résumé par saison (note moyenne ± écart-type, nb épisodes) :")
@printf("  %-8s  %-12s  %-12s  %s\n", "Saison", "Note moy.", "Écart-type", "Nb épisodes")
println("  ", "-"^52)

# groupby : regroupe les lignes par valeur de la colonne :Season
# combine  : applique des fonctions d'agrégation sur chaque groupe
season_summary = combine(groupby(op_eps, :Season),
    :Rating => (x -> mean(skipmissing(x)))   => :Moy,
    :Rating => (x -> std(skipmissing(x)))    => :Std,
    :Rating => (x -> length(collect(skipmissing(x)))) => :N,
)
sort!(season_summary, :Season)

for row in eachrow(season_summary)
    @printf("  %-8d  %-12.3f  %-12.3f  %d\n",
            row.Season, row.Moy, row.Std, row.N)
end

# ── Zoom One Piece : note globale ─────────────────────────────────────────────
println("\n\n", "="^70)
println("DONNÉES ONE PIECE — all-series-ep-average.csv")
println("="^70)

avg_df = CSV.read(joinpath(DATA_DIR, "all-series-ep-average.csv"), DataFrame;
                  missingstring=["NA","","N/A"], header=false)
# Colonnes sans en-tête : Code, Title, GlobalRating, RatingCount, NbEps, EpMean
rename!(avg_df, [:Code, :Title, :GlobalRating, :RatingCount, :NbEps, :EpMean])
op_avg = filter(row -> row.Code == OP_CODE, avg_df)
println("  ", op_avg)

println("\n✔  Exploration terminée.")
