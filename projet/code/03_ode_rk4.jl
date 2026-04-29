# 03_ode_rk4.jl — Modèle ODE intra-saga + prédiction saga 10
# Ch4 : ODE de type retour à la moyenne, résolution RK4 codé à la main
# Ch2 : discussion erreur de troncature RK4 vs Euler, seed fixée
# Ch3 : série temporelle des paramètres K et λ sur les 9 sagas

import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
using CSV, DataFrames, Statistics, Plots, Printf
import Random; Random.seed!(42)

DATA_DIR    = joinpath(@__DIR__, "..", "data")
RESULTS_DIR = joinpath(@__DIR__, "..", "results")
OP_CODE     = "tt0388629"

# ── Chargement des données ────────────────────────────────────────────────────
ep_df = CSV.read(joinpath(DATA_DIR, "all-episode-ratings.csv"), DataFrame;
                 missingstring=["NA","","N/A"])
op    = filter(row -> !ismissing(row.Code) && row.Code == OP_CODE, ep_df)
sort!(op, :Episode)

# Découpage en sagas (même table qu'en étape 2)
const SAGAS = [
    (1, "East Blue",         1,   61),
    (2, "Alabasta",         62,  135),
    (3, "Sky Island",      136,  206),
    (4, "Water 7",         207,  325),
    (5, "Thriller Bark",   326,  384),
    (6, "Summit War",      385,  516),
    (7, "Fish-Man Island", 517,  574),
    (8, "Dressrosa",       575,  746),
    (9, "Quatre Empereurs",747,  883),
]
op[!, :SagaId]   = zeros(Int, nrow(op))
op[!, :SagaName] = fill("", nrow(op))
for (id, name, a, b) in SAGAS
    mask = (op.Episode .>= a) .& (op.Episode .<= b)
    op[mask, :SagaId]   .= id
    op[mask, :SagaName] .= name
end

# ── 1. IMPLÉMENTATION RK4 (Ch4) ───────────────────────────────────────────────
#
# Pour dy/dt = f(t, y) on fait avancer la solution d'un pas h :
#
#   k₁ = f(tₙ,         yₙ)                ← pente au début du pas
#   k₂ = f(tₙ + h/2,   yₙ + h/2 · k₁)    ← pente au milieu (1ʳᵉ approx.)
#   k₃ = f(tₙ + h/2,   yₙ + h/2 · k₂)    ← pente au milieu (2ᵉ approx.)
#   k₄ = f(tₙ + h,     yₙ + h · k₃)      ← pente à la fin
#   yₙ₊₁ = yₙ + (h/6)·(k₁ + 2k₂ + 2k₃ + k₄)
#
# Erreur locale : O(h⁵) ; erreur globale : O(h⁴)
# Euler explicite n'a qu'une erreur O(h²) → RK4 est ~h³ fois plus précis
# pour le même coût par pas. (Ch2 : erreurs numériques)
#
function rk4_step(f, t::Float64, y::Float64, h::Float64)::Float64
    k1 = f(t,       y)
    k2 = f(t + h/2, y + (h/2) * k1)
    k3 = f(t + h/2, y + (h/2) * k2)
    k4 = f(t + h,   y + h      * k3)
    return y + (h / 6) * (k1 + 2k2 + 2k3 + k4)
end

# Intègre de t0 à t_end en n_steps pas, retourne (ts, ys)
function rk4_integrate(f, t0::Float64, y0::Float64,
                        t_end::Float64, n_steps::Int)
    h  = (t_end - t0) / n_steps
    ts = Vector{Float64}(undef, n_steps + 1)
    ys = Vector{Float64}(undef, n_steps + 1)
    ts[1] = t0; ys[1] = y0
    for i in 1:n_steps
        ys[i+1] = rk4_step(f, ts[i], ys[i], h)
        ts[i+1] = ts[i] + h
    end
    return ts, ys
end

# ── 2. ODE INTRA-SAGA : modèle de retour à la moyenne ────────────────────────
#
# À l'intérieur d'une saga les notes oscillent autour d'un niveau K
# (qualité intrinsèque de la saga). Le modèle est :
#
#   dR/dt = -λ · (R(t) - K)
#
# Solution analytique : R(t) = K + (R₀ - K) · exp(-λt)
#   → converge exponentiellement vers K
#   → K = "attracteur" = qualité de fond de la saga
#   → λ = vitesse de retour (grand λ : notes stables ; petit λ : inertie forte)
#
# On estime K et λ en minimisant la MSE entre la courbe RK4 et les notes réelles.
# Le "temps" t est l'indice local dans la saga (0, 1, 2, …, N-1).
#
ode_intra(λ, K) = (t, R) -> -λ * (R - K)

# Erreur MSE de l'ODE sur une série de notes (λ et K fixés)
function mse_intra(ratings::Vector{Float64}, λ::Float64, K::Float64)::Float64
    n   = length(ratings)
    n < 2 && return 0.0
    t0  = 0.0; t_end = Float64(n - 1)
    _, ys = rk4_integrate(ode_intra(λ, K), t0, ratings[1], t_end, n - 1)
    return mean((ys[1:n] .- ratings).^2)
end

# Grille de recherche pour λ et K
λ_grid = 0.005:0.005:0.5
K_grid = 7.0:0.05:8.5

function fit_saga(ratings::Vector{Float64})
    best_mse = Inf
    best_λ   = NaN
    best_K   = NaN
    for λ in λ_grid, K in K_grid
        m = mse_intra(ratings, λ, K)
        if m < best_mse
            best_mse = m; best_λ = λ; best_K = K
        end
    end
    return best_λ, best_K, best_mse
end

# ── 3. Ajustement sur chaque saga ─────────────────────────────────────────────
println("Ajustement ODE intra-saga...")
@printf("%-4s  %-22s  %6s  %7s  %7s  %9s\n",
        "Saga", "Nom", "NbEps", "λ_opt", "K_opt", "MSE")
println("-"^60)

fit_results = DataFrame(SagaId=Int[], SagaName=String[],
                         Lambda=Float64[], K=Float64[], MSE=Float64[])

for (id, name, ep_a, ep_b) in SAGAS
    sub = filter(row -> row.SagaId == id, op)
    ratings = Float64.(sub.Rating)
    λ_opt, K_opt, mse = fit_saga(ratings)
    push!(fit_results, (id, name, λ_opt, K_opt, mse))
    @printf("%-4d  %-22s  %6d  %7.3f  %7.3f  %9.6f\n",
            id, name, length(ratings), λ_opt, K_opt, mse)
end

# ── 4. PRÉDICTION SAGA 10 ─────────────────────────────────────────────────────
# On extrapole l'attracteur K à la saga 10 via régression linéaire sur K₁…K₉
# (même principe qu'au Ch3 : série temporelle des paramètres)
t_data = Float64.(fit_results.SagaId)
K_data = fit_results.K

β_K = cov(t_data, K_data) / var(t_data)
α_K = mean(K_data) - β_K * mean(t_data)
K_pred10 = α_K + β_K * 10.0

# λ prédit : moyenne des λ passés (pas de tendance claire, on suppose stable)
λ_pred10 = mean(fit_results.Lambda)

@printf("\nTendance de K : K̂(t) = %.4f + %.4f·t\n", α_K, β_K)
@printf("K prédit pour saga 10 : %.4f\n", K_pred10)
@printf("λ moyen utilisé       : %.4f\n", λ_pred10)

# Simulation : profil d'épisodes prédit pour la saga 10
# On suppose ~120 épisodes (médiane des sagas précédentes)
N_EPS_PRED = 120

# Déviation initiale (R₀ - K) observée sur chaque saga
# → donne une idée de "comment une saga One Piece démarre par rapport à son attracteur"
R0_each = Float64[]
for (id, _, ep_a, ep_b) in SAGAS
    sub = filter(row -> row.SagaId == id, op)
    push!(R0_each, sub.Rating[1])
end
δ_mean = mean(R0_each .- fit_results.K)   # déviation initiale moyenne
R0_pred = K_pred10 + δ_mean              # R₀ réaliste = K_prédit + décalage typique
@printf("Déviation initiale moyenne R₀ - K : %.4f\n", δ_mean)
@printf("R₀ utilisé pour saga 10 : %.4f\n", R0_pred)

t_p, R_p = rk4_integrate(ode_intra(λ_pred10, K_pred10),
                           0.0, R0_pred, Float64(N_EPS_PRED - 1), N_EPS_PRED - 1)

mean_pred = mean(R_p)
@printf("\nNote moyenne prédite saga 10 : %.4f\n", mean_pred)
@printf("Seuil succès 8.0 : %s\n",
        mean_pred >= 8.0 ? "ATTEINT ✓" : "NON atteint ✗")

# Fourchette : ±1 écart-type moyen des résidus
σ_residus = mean(sqrt.(fit_results.MSE))
@printf("Intervalle ±1σ_résidus : [%.4f, %.4f]\n",
        mean_pred - σ_residus, mean_pred + σ_residus)

# ── 5. Sauvegarde ─────────────────────────────────────────────────────────────
CSV.write(joinpath(RESULTS_DIR, "ode_fit_results.csv"), fit_results)

pred_df = DataFrame(
    Saga       = [10],
    Nom        = ["Wano Country (prédit)"],
    K_predit   = [K_pred10],
    Lambda     = [λ_pred10],
    Note_pred  = [mean_pred],
    IC_bas     = [mean_pred - σ_residus],
    IC_haut    = [mean_pred + σ_residus],
)
CSV.write(joinpath(RESULTS_DIR, "prediction_saga10.csv"), pred_df)
println("\nFichiers sauvegardés dans results/")

# ── 6. GRAPHIQUES ─────────────────────────────────────────────────────────────

# -- 6a : Courbe ODE sur la saga 9 (la plus récente, la plus pertinente) -------
saga9    = filter(row -> row.SagaId == 9, op)
r9       = Float64.(saga9.Rating)
λ9 = fit_results[fit_results.SagaId .== 9, :Lambda][1]
K9 = fit_results[fit_results.SagaId .== 9, :K][1]

t9_ode, R9_ode = rk4_integrate(ode_intra(λ9, K9),
                                 0.0, r9[1], Float64(length(r9)-1), length(r9)-1)

p1 = plot(1:length(r9), r9;
    alpha     = 0.5, color = :steelblue, linewidth = 1,
    label     = "Notes réelles",
    xlabel    = "Épisode (dans la saga)",
    ylabel    = "Note IMDb",
    title     = "Saga 9 — Quatre Empereurs : ODE RK4 ajustée",
    legend    = :bottomright, size = (800, 400),
)
plot!(p1, t9_ode .+ 1, R9_ode;
    linewidth = 2.5, color = :orange,
    label = @sprintf("ODE RK4 (λ=%.3f, K=%.2f)", λ9, K9),
)
hline!(p1, [K9];
    linestyle = :dash, color = :red, linewidth = 1.5,
    label = "Attracteur K",
)
savefig(p1, joinpath(RESULTS_DIR, "ode_saga9.png"))

# -- 6b : Évolution de K par saga + prédiction saga 10 -------------------------
t_all = vcat(t_data, [10.0])
K_all = vcat(K_data, [K_pred10])

p2 = scatter(t_data, K_data;
    marker    = :circle, markersize = 7, color = :royalblue,
    label     = "K ajusté par saga",
    xlabel    = "Numéro de saga",
    ylabel    = "Attracteur K (qualité de fond)",
    title     = "Évolution de l'attracteur K — One Piece",
    xticks    = 1:10, legend = :bottomright, size = (800, 400),
)
t_line = 1.0:0.1:10.0
plot!(p2, collect(t_line), α_K .+ β_K .* t_line;
    linestyle = :dash, color = :gray, linewidth = 1.5,
    label = "Tendance linéaire",
)
scatter!(p2, [10.0], [K_pred10];
    marker    = :star5, markersize = 12, color = :red,
    label = @sprintf("Saga 10 : K = %.3f", K_pred10),
)
hline!(p2, [8.0];
    linestyle = :dot, color = :red, linewidth = 1.5,
    label = "Seuil succès",
)
savefig(p2, joinpath(RESULTS_DIR, "ode_K_evolution.png"))

# -- 6c : Profil prédit des épisodes de la saga 10 -----------------------------
p3 = plot(t_p .+ 1, R_p;
    linewidth = 2.5, color = :darkorange,
    label     = @sprintf("ODE RK4 (K=%.3f, λ=%.3f)", K_pred10, λ_pred10),
    xlabel    = "Épisode (dans la saga 10 prédite)",
    ylabel    = "Note IMDb prédite",
    title     = "Saga 10 (Wano) — Profil prédit par ODE",
    legend    = :topright, size = (800, 400),
)
hline!(p3, [mean_pred];
    color = :darkorange, linestyle = :dash, linewidth = 1.5,
    label = @sprintf("Moyenne prédite = %.3f", mean_pred),
)
hline!(p3, [8.0];
    linestyle = :dot, color = :red, linewidth = 1.5,
    label = "Seuil succès (8.0)",
)
# Bande d'incertitude
plot!(p3, t_p .+ 1, fill(mean_pred + σ_residus, length(t_p));
    fillrange = fill(mean_pred - σ_residus, length(t_p)),
    alpha = 0.15, color = :orange, label = "±1σ résidus", linewidth = 0,
)
savefig(p3, joinpath(RESULTS_DIR, "ode_saga10_prediction.png"))

println("\n✔  Modèle ODE terminé. Graphiques → results/")
