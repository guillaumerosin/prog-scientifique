# 05_adf_arima.jl — Test ADF + modèle ARIMA
# ADF  : vérifie la stationnarité des 883 notes d'épisodes
# ARIMA: prédit la note moyenne de la saga 10 à partir des 9 sagas

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using CSV, DataFrames, Statistics, Printf, LinearAlgebra

DATA_DIR    = joinpath(@__DIR__, "..", "data")
RESULTS_DIR = joinpath(@__DIR__, "..", "results")

OP_CODE = "tt0388629"

# ── 1. Chargement des données ──────────────────────────────────────────────────
ep_df = CSV.read(joinpath(DATA_DIR, "all-episode-ratings.csv"), DataFrame;
                 missingstring=["NA","","N/A"])
op = filter(row -> !ismissing(row.Code) && row.Code == OP_CODE, ep_df)
sort!(op, :Episode)
y_ep = Float64.(op.Rating)   # 883 notes d'épisodes

sagas_df = CSV.read(joinpath(RESULTS_DIR, "one_piece_sagas.csv"), DataFrame)
y_saga = Float64.(sagas_df.MoySimple)   # 9 notes de sagas

# ── 2. Test ADF (Augmented Dickey-Fuller) ─────────────────────────────────────
# H₀ : la série a une racine unitaire (non stationnaire)
# H₁ : la série est stationnaire
#
# Régression ADF(1) avec constante :
#   Δy_t = α + γ·y_{t-1} + δ·Δy_{t-1} + ε_t
#
# On teste H₀ : γ = 0  vs  H₁ : γ < 0
# via le t-stat de γ comparé aux valeurs critiques de MacKinnon (1994)

function adf_test(y::Vector{Float64}; lags::Int=1)
    n  = length(y)
    dy = diff(y)                        # Δy_t = y_t - y_{t-1}

    # Construction de la matrice de régression
    # On a besoin de lags+1 observations supplémentaires
    T  = length(dy) - lags              # nombre d'observations utilisées
    Y  = dy[lags+1:end]                 # variable dépendante : Δy_t
    X  = ones(T, lags + 2)              # constante + y_{t-1} + Δy_{t-1..lag}
    X[:, 2] = y[lags+1:end-1]          # y_{t-1}
    for l in 1:lags
        X[:, 2+l] = dy[lags+1-l:end-l] # Δy_{t-l}
    end

    # OLS : β = (X'X)⁻¹ X'Y
    beta   = (X' * X) \ (X' * Y)
    resid  = Y - X * beta
    sigma2 = sum(resid .^ 2) / (T - size(X, 2))
    se     = sqrt.(diag(sigma2 * inv(X' * X)))

    gamma  = beta[2]                    # coefficient de y_{t-1}
    t_stat = gamma / se[2]             # t-statistique de γ

    # Valeurs critiques MacKinnon (1994) — modèle avec constante, n > 500
    cv = Dict("1%" => -3.43, "5%" => -2.86, "10%" => -2.57)

    return (t_stat=t_stat, gamma=gamma, cv=cv)
end

println("\n", "="^60)
println("TEST ADF — Notes par épisode One Piece (n = $(length(y_ep)))")
println("="^60)

res = adf_test(y_ep; lags=1)
@printf("  t-statistique ADF : %.4f\n", res.t_stat)
@printf("  Valeurs critiques  : 1%% = %.2f  |  5%% = %.2f  |  10%% = %.2f\n",
        res.cv["1%"], res.cv["5%"], res.cv["10%"])

if res.t_stat < res.cv["1%"]
    println("  → Rejet de H₀ à 1% : série STATIONNAIRE")
    d_order = 0
elseif res.t_stat < res.cv["5%"]
    println("  → Rejet de H₀ à 5% : série STATIONNAIRE")
    d_order = 0
elseif res.t_stat < res.cv["10%"]
    println("  → Rejet de H₀ à 10% : série STATIONNAIRE (limite)")
    d_order = 0
else
    println("  → Échec de rejet de H₀ : série NON STATIONNAIRE → d = 1")
    d_order = 1
end

# ── 3. ARIMA(1, d, 0) sur les 9 sagas ────────────────────────────────────────
# AR(1) sur la série différenciée d fois
# Prédiction de la saga 10 par back-transformation

println("\n", "="^60)
println("ARIMA(1,$(d_order),0) — Notes moyennes par saga (n = $(length(y_saga)))")
println("="^60)

function arima_predict(y::Vector{Float64}, d::Int)
    # Différenciation d fois
    yd = copy(y)
    for _ in 1:d
        yd = diff(yd)
    end

    n  = length(yd)
    # AR(1) : yd_t = α + φ·yd_{t-1} + ε_t  (OLS)
    Y  = yd[2:end]
    X  = hcat(ones(n-1), yd[1:end-1])
    beta = (X' * X) \ (X' * Y)
    alpha, phi = beta[1], beta[2]

    resid  = Y - X * beta
    sigma  = std(resid)

    @printf("  AR(1) : α = %.4f  |  φ = %.4f\n", alpha, phi)
    @printf("  Écart-type résidus : %.4f\n", sigma)

    # Prédiction un pas en avant sur la série différenciée
    yd_pred = alpha + phi * yd[end]

    # Back-transformation : on réintègre d fois
    pred = yd_pred
    if d == 1
        pred = pred + y[end]   # y_{n+1} = Δy_{n+1} + y_n
    end

    # IC à 95% (±1.96σ sur la série différenciée, retransformé)
    ic_low  = pred - 1.96 * sigma
    ic_high = pred + 1.96 * sigma

    return (pred=pred, ic_low=ic_low, ic_high=ic_high, sigma=sigma)
end

res_arima = arima_predict(y_saga, d_order)
@printf("\n  Prédiction saga 10 : %.4f\n", res_arima.pred)
@printf("  IC 95%%             : [%.3f ; %.3f]\n", res_arima.ic_low, res_arima.ic_high)
@printf("  Seuil succès (8.0) : %s\n",
        res_arima.ic_low >= 8.0 ? "DANS l'IC" : "HORS de l'IC")
