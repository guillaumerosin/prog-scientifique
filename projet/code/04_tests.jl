# 04_tests.jl — Tests statistiques (Ch5)
# Test 1 : Mann-Whitney — One Piece vs groupe de séries longues à succès
# Test 2 : Wilcoxon signed-rank — la prédiction est-elle compatible avec 8.0 ?
# Test 3 : Bootstrap — intervalle de confiance sur la note moyenne de la saga 10

import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
using CSV, DataFrames, Statistics, HypothesisTests, Plots, StatsPlots, Printf
import Random
Random.seed!(42)

DATA_DIR    = joinpath(@__DIR__, "..", "data")
RESULTS_DIR = joinpath(@__DIR__, "..", "results")
OP_CODE     = "tt0388629"

# ── Chargement des données ────────────────────────────────────────────────────
ts_op  = CSV.read(joinpath(RESULTS_DIR, "one_piece_sagas.csv"), DataFrame)
pred   = CSV.read(joinpath(RESULTS_DIR, "prediction_saga10.csv"), DataFrame)
ts_all = CSV.read(joinpath(DATA_DIR, "top-seasons-full.csv"), DataFrame)
ep_df  = CSV.read(joinpath(DATA_DIR, "all-episode-ratings.csv"), DataFrame;
                  missingstring=["NA","","N/A"])

# ── PRÉPARATION DES DONNÉES ───────────────────────────────────────────────────

# Notes de saga d'One Piece (nos 9 points historiques)
op_saga_means = ts_op.MoySimple          # Vector{Float64}, length 9

# Groupe de référence : séries avec ≥ 5 sagas dans top-seasons-full.csv
# (même statut "longue série de qualité" qu'One Piece)
# On exclut One Piece lui-même.
n_sagas_by_series = combine(groupby(ts_all, :Key), nrow => :N)
long_series_keys  = filter(r -> r.N >= 5 && r.Key != OP_CODE,
                             n_sagas_by_series).Key

ref_sagas = filter(r -> r.Key in long_series_keys, ts_all)
ref_means = ref_sagas[!, Symbol("Rating Mean")]   # toutes les moyennes de saga

@printf("One Piece : %d sagas  (moyenne = %.3f)\n",
        length(op_saga_means), mean(op_saga_means))
@printf("Groupe référence : %d sagas issues de %d séries longues\n",
        length(ref_means), length(long_series_keys))
@printf("  Médiane référence = %.3f\n", median(ref_means))

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Mann-Whitney (Wilcoxon rank-sum, deux échantillons)
# ═══════════════════════════════════════════════════════════════════════════════
#
# H₀ : La distribution des notes de saga d'One Piece est identique
#       à celle des séries longues du groupe de référence.
# H₁ : Les deux distributions sont différentes (bilatéral).
#
# Choix du test non-paramétrique : les échantillons sont petits (9 points pour
# One Piece) et on ne peut pas supposer la normalité des notes de saga.
#
# MannWhitneyUTest(x, y) de HypothesisTests.jl
#   → calcule la statistique U et retourne un p-value exact (ou approximé si n grand)
#
println("\n", "═"^60)
println("TEST 1 — Mann-Whitney : One Piece vs groupe de référence")
println("═"^60)

mw = MannWhitneyUTest(op_saga_means, Float64.(ref_means))
println(mw)
p1 = pvalue(mw)
@printf("p-value (bilatérale) = %.4f\n", p1)
@printf("Conclusion : %s\n",
        p1 < 0.05 ?
        "Rejet de H₀ — One Piece se distingue significativement du groupe (α=0.05)" :
        "Non-rejet de H₀ — One Piece appartient statistiquement au même groupe (α=0.05)")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Wilcoxon signed-rank (un seul échantillon)
# ═══════════════════════════════════════════════════════════════════════════════
#
# H₀ : La médiane des notes de saga d'One Piece = 8.0 (seuil succès)
# H₁ : La médiane < 8.0 (test unilatéral gauche)
#
# SignedRankTest(x, μ₀) teste si la médiane de x vaut μ₀.
# On utilise le test unilatéral gauche : pvalue(test, tail=:left)
#
SEUIL = 8.0
println("\n", "═"^60)
println("TEST 2 — Wilcoxon signed-rank : médiane sagas One Piece vs $SEUIL")
println("═"^60)

sr = SignedRankTest(op_saga_means .- SEUIL)   # teste si la médiane des (x - 8) = 0
println(sr)
p2_left = pvalue(sr, tail=:left)
@printf("p-value (unilatérale gauche, H₁: médiane < %.1f) = %.4f\n", SEUIL, p2_left)
@printf("Conclusion : %s\n",
        p2_left < 0.05 ?
        "La médiane des sagas d'One Piece est significativement < 8.0 (α=0.05)" :
        "Pas de preuve que la médiane soit < 8.0 (α=0.05)")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Bootstrap sur la prédiction de la saga 10
# ═══════════════════════════════════════════════════════════════════════════════
#
# Principe : on rééchantillonne avec remise B fois dans les épisodes de la
# saga 9 (la plus récente, notre meilleure approximation de saga 10).
# Chaque tirage simule un jeu possible de 120 notes pour la saga 10.
# → Distribution bootstrap des moyennes possibles → IC et p-value.
#
# C'est une approche non-paramétrique : on ne suppose pas de loi particulière.
#
saga9_ratings = begin
    op = filter(r -> !ismissing(r.Code) && r.Code == OP_CODE, ep_df)
    sort!(op, :Episode)
    # Épisodes saga 9 : 747 → 883
    Float64.(filter(r -> r.Episode >= 747 && r.Episode <= 883, op).Rating)
end

N_BOOT    = 10_000   # nombre de rééchantillonnages
N_SIM_EPS = 120      # taille supposée de la saga 10

# Tirages bootstrap : chaque ligne = une saga 10 simulée
boot_means = [mean(rand(saga9_ratings, N_SIM_EPS)) for _ in 1:N_BOOT]

# IC à 95 %
ic_bas  = quantile(boot_means, 0.025)
ic_haut = quantile(boot_means, 0.975)

# p-value bootstrap : proportion de simulations avec moyenne ≥ 8.0
p3_boot = mean(boot_means .>= SEUIL)

println("\n", "═"^60)
println("TEST 3 — Bootstrap (B=$N_BOOT, saga 10 = $N_SIM_EPS épisodes)")
println("═"^60)
@printf("Moyenne des moyennes bootstrap : %.4f\n", mean(boot_means))
@printf("IC bootstrap à 95 %%           : [%.4f, %.4f]\n", ic_bas, ic_haut)
@printf("P(moyenne ≥ %.1f)              : %.4f\n", SEUIL, p3_boot)
@printf("Conclusion : %s\n",
        p3_boot < 0.05 ?
        "Improbable qu'One Piece atteigne 8.0 de moyenne (p < 5 %)" :
        "Il est plausible qu'One Piece atteigne 8.0 de moyenne")

# ═══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ═══════════════════════════════════════════════════════════════════════════════
println("\n", "═"^60)
println("SYNTHÈSE DES TESTS")
println("═"^60)
@printf("Test 1 (Mann-Whitney)  p = %.4f → %s\n",
        p1, p1 < 0.05 ? "One Piece ≠ groupe référence" : "One Piece ∈ groupe référence")
@printf("Test 2 (Signed-rank)   p = %.4f → %s\n",
        p2_left, p2_left < 0.05 ? "Médiane saga < 8.0 confirmée" : "Médiane saga ≈ 8.0 plausible")
@printf("Test 3 (Bootstrap)     P(≥8) = %.4f → %s\n",
        p3_boot, p3_boot < 0.05 ? "Peu probable d'atteindre 8.0" : "8.0 reste plausible")

# Sauvegarde
summary_df = DataFrame(
    Test        = ["Mann-Whitney (1 vs ref)", "Wilcoxon signed-rank", "Bootstrap P(≥8)"],
    H0          = ["distributions égales", "médiane = 8.0", "P(moy ≥ 8.0)"],
    Statistique = [mw.U, sr.W, p3_boot],
    p_value     = [p1, p2_left, p3_boot],
    Rejet_H0    = [p1 < 0.05, p2_left < 0.05, p3_boot < 0.05],
)
CSV.write(joinpath(RESULTS_DIR, "tests_statistiques.csv"), summary_df)

# ── GRAPHIQUES ────────────────────────────────────────────────────────────────

# Distribution bootstrap
p_boot = histogram(boot_means;
    bins      = 60,
    color     = :steelblue,
    alpha     = 0.7,
    label     = "Moyennes bootstrap (B=$N_BOOT)",
    xlabel    = "Note moyenne simulée (saga 10)",
    ylabel    = "Fréquence",
    title     = "Bootstrap — Distribution des notes possibles pour la saga 10",
    legend    = :topright,
    size      = (800, 420),
)
vline!(p_boot, [SEUIL];
    color = :red, linestyle = :dash, linewidth = 2,
    label = "Seuil succès (8.0)",
)
vline!(p_boot, [mean(boot_means)];
    color = :orange, linestyle = :solid, linewidth = 2,
    label = @sprintf("Moyenne bootstrap = %.3f", mean(boot_means)),
)
vline!(p_boot, [ic_bas, ic_haut];
    color = :gray, linestyle = :dot, linewidth = 1.5,
    label = @sprintf("IC 95 %% [%.3f, %.3f]", ic_bas, ic_haut),
)
savefig(p_boot, joinpath(RESULTS_DIR, "bootstrap_saga10.png"))

# Boîtes à moustaches : One Piece vs référence
p_box = boxplot(
    fill("Référence\n(séries longues)", length(ref_means)),
    Float64.(ref_means);
    color    = :steelblue,
    alpha    = 0.7,
    label    = "Groupe référence",
    ylabel   = "Note moyenne par saga",
    title    = "One Piece vs séries longues — notes par saga",
    legend   = :topright,
    size     = (700, 480),
    outliers = true,
)
boxplot!(p_box,
    fill("One Piece\n(sagas 1-9)", length(op_saga_means)),
    op_saga_means;
    color  = :orange,
    alpha  = 0.8,
    label  = "One Piece",
)
hline!(p_box, [SEUIL];
    color = :red, linestyle = :dot, linewidth = 1.5,
    label = "Seuil 8.0",
)
savefig(p_box, joinpath(RESULTS_DIR, "boxplot_comparison.png"))

println("\n✔  Tests statistiques terminés. Graphiques → results/")
