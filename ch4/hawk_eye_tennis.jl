"""
Hawk-Eye for Tennis - Exercice Chapitre 4
Scientific Programming - J-S Lerat (HEH)

Simulation de la trajectoire d'une balle de tennis avec :
- Effet Magnus (M)
- Résistance de l'air (D)
- Gravité

Implémentation RK4 en Julia
"""

# ============================================================
# Mes constantes physiques et les dimensions du terrain
# ============================================================

const ρ  = 1.23          # kg/m³ - densité de l'air au niveau de la mer
const R  = 0.0335        # m     - rayon de la balle (~3.35 cm)
const m  = 0.058         # kg    - masse de la balle (58 g, corrigé depuis 0.58 g)
const g  = 9.81          # m/s²  - gravité

# Dimensions du terrain (demi-terrain depuis le joueur)
const L_court     = 11.89        # m - longueur demi-terrain (x1)
const W_court_half = 4.115       # m - demi-largeur du terrain (x2), total = 8.23 m
const H_net       = 0.914        # m - hauteur du filet au centre
const H_net_side  = 1.07         # m - hauteur du filet sur les côtés (~1.07 m)
const X_net       = 11.89        # m - position du filet (x1)
const H_player    = 1.0          # m - hauteur de frappe approximative (waist level)

# ============================================================
# Calcul des coefficients aérodynamiques
# ============================================================

"""
    compute_q(dx_dt, omega)

Calcule le ratio spin q = |∂x| / (R |ω|)
Pour ω = 0 (flat), q → ∞
"""
function compute_q(v::Vector{Float64}, omega::Vector{Float64})
    v_norm = norm(v)
    ω_norm = norm(omega)
    if ω_norm < 1e-10
        return Inf  # flat shot → q → ∞
    end
    return v_norm / (R * ω_norm)
end

"""
    compute_CM(q)

Coefficient de Magnus.
Pour flat (q → ∞) : CM → 0
"""
function compute_CM(q::Float64)
    if isinf(q)
        return 0.0
    end
    return 1.0 / (2.022 + 0.981 * q)
end

"""
    compute_CD(q)

Coefficient de traînée.
Pour flat (q → ∞) : CD → 0.508
"""
function compute_CD(q::Float64)
    if isinf(q)
        return 0.508
    end
    inner = 1.0 / (22.053 + 4.196 * q^(5/2))
    return 0.508 + inner^(2/5)
end

# ============================================================
# Produit vectoriel 3D
# ============================================================

function cross3(a::Vector{Float64}, b::Vector{Float64})
    return [
        a[2]*b[3] - a[3]*b[2],
        a[3]*b[1] - a[1]*b[3],
        a[1]*b[2] - a[2]*b[1]
    ]
end

norm(v::Vector{Float64}) = sqrt(sum(v.^2))

# ============================================================
# Forces : Magnus M et traînée D
# ============================================================

"""
    magnus_force(v, omega)

M = (1/2) ρ CM π R² |∂x|² (ω/|ω|) × (∂x/|∂x|)
"""
function magnus_force(v::Vector{Float64}, omega::Vector{Float64})
    v_norm = norm(v)
    ω_norm = norm(omega)
    if ω_norm < 1e-10 || v_norm < 1e-10
        return [0.0, 0.0, 0.0]
    end
    q  = compute_q(v, omega)
    CM = compute_CM(q)
    ω_hat = omega / ω_norm
    v_hat = v / v_norm
    return 0.5 * ρ * CM * π * R^2 * v_norm^2 * cross3(ω_hat, v_hat)
end

"""
    drag_force(v, omega)

D = -(1/2) ρ CD π R² |∂x|² (∂x/|∂x|)
"""
function drag_force(v::Vector{Float64}, omega::Vector{Float64})
    v_norm = norm(v)
    if v_norm < 1e-10
        return [0.0, 0.0, 0.0]
    end
    q  = compute_q(v, omega)
    CD = compute_CD(q)
    v_hat = v / v_norm
    return -0.5 * ρ * CD * π * R^2 * v_norm^2 * v_hat
end

# ============================================================
# Problème de Cauchy — Équation du mouvement
# ============================================================
# État: u = [x1, x2, x3, v1, v2, v3]  (position + vitesse)
# u'(t) = f(t, u)

"""
    f_ball(t, u, omega)

Définit le membre droit du problème de Cauchy :
    m ∂²x = (0, 0, -mg) + D + M

Réécrit comme système du 1er ordre :
    dx/dt = v
    dv/dt = (1/m) * [(0,0,-mg) + D + M]
"""
function f_ball(t::Float64, u::Vector{Float64}, omega::Vector{Float64})
    pos = u[1:3]   # x = (x1, x2, x3)
    vel = u[4:6]   # v = ∂x

    gravity = [0.0, 0.0, -m * g]
    D = drag_force(vel, omega)
    M = magnus_force(vel, omega)

    accel = (gravity + D + M) / m   # ∂²x = F/m

    return [vel[1], vel[2], vel[3],
            accel[1], accel[2], accel[3]]
end

# ============================================================
# Solveur RK4 (Runge-Kutta classique, 4 étapes)
# ============================================================
# Butcher tableau RK4 :
#  0   |
# 1/2  | 1/2
# 1/2  |  0  1/2
#  1   |  0   0   1
#      | 1/6 1/3 1/3 1/6

"""
    rk4_step(f, t, u, h, omega)

Un pas de RK4 classique.
"""
function rk4_step(f, t::Float64, u::Vector{Float64}, h::Float64, omega::Vector{Float64})
    k1 = f(t,           u,                 omega)
    k2 = f(t + h/2,     u + (h/2) * k1,   omega)
    k3 = f(t + h/2,     u + (h/2) * k2,   omega)
    k4 = f(t + h,       u + h * k3,        omega)
    return u + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
end

"""
    simulate(u0, omega; h=0.001, t_max=5.0)

Simule la trajectoire de la balle jusqu'à ce qu'elle touche le sol (x3 ≤ 0)
ou dépasse le temps maximum.

Retourne (ts, xs) : vecteur des temps et matrice des états.
"""
function simulate(u0::Vector{Float64}, omega::Vector{Float64};
                  h::Float64=0.001, t_max::Float64=5.0)
    ts = Float64[0.0]
    us = Vector{Float64}[copy(u0)]

    t = 0.0
    u = copy(u0)

    while t < t_max
        u_new = rk4_step(f_ball, t, u, h, omega)
        t += h

        push!(ts, t)
        push!(us, copy(u_new))

        u = u_new

        # Arrêt si la balle touche le sol
        if u[3] <= 0.0
            break
        end
    end

    return ts, us
end

# ============================================================
# Conditions initiales
# ============================================================
# Position initiale du joueur (origine du terrain)
# Le joueur frappe depuis x1=0, x2=0, x3=H_player
# Angle initial θ = 10° entre x1 et x3 → vers le bas = -10°

"""
    initial_conditions(speed, shot_type; angle_deg=-10.0)

Retourne [x0; v0] pour les trois types de coups.
- speed : vitesse initiale (m/s)
- shot_type : :flat, :topspin, :slice
- angle_deg : angle par rapport à l'horizontale (négatif = vers le bas)

Position initiale : (0, 0, H_player)
Vitesse : dans le plan (x1, x3) avec angle θ
    v1 = speed * cos(θ)
    v3 = speed * sin(θ)   (négatif pour frapper vers le bas)

Spin ω (autour de l'axe x2 = droite-gauche, règle de la main droite) :
  - flat    : ω = (0, 0, 0)
  - topspin : ω_y = +|ω|   (balle tourne vers l'avant-bas)
  - slice   : ω_y = -|ω|   (balle tourne vers l'avant-haut)
"""
function initial_conditions(speed::Float64, shot_type::Symbol;
                            angle_deg::Float64=-10.0,
                            omega_mag::Float64=30.0)
    θ = deg2rad(angle_deg)

    x0 = [0.0, 0.0, H_player]     # position initiale
    v0 = [speed * cos(θ),          # v1 : vers le filet
          0.0,                      # v2 : pas de déviation latérale
          speed * sin(θ)]           # v3 : composante verticale (négative si θ<0)

    if shot_type == :flat
        omega = [0.0, 0.0, 0.0]
    elseif shot_type == :topspin
        omega = [0.0, omega_mag, 0.0]   # ωi > 0 (i = axe correspondant à x3)
    elseif shot_type == :slice
        omega = [0.0, -omega_mag, 0.0]  # ωi < 0
    else
        error("Type de coup inconnu : $shot_type")
    end

    u0 = [x0; v0]
    return u0, omega
end

# ============================================================
# Validation : la balle est-elle "dans le court" ?
# ============================================================

"""
    is_on_court(us)

Vérifie que la balle :
1. Passe au-dessus du filet (x1 ≈ 11.89 m, x3 ≥ hauteur du filet)
2. Rebondit dans le demi-terrain adverse (11.89 ≤ x1 ≤ 23.78)
3. Reste dans la largeur du court (|x2| ≤ 4.115 m)

Retourne (valid, x_bounce, clears_net)
"""
function is_on_court(ts::Vector{Float64}, us::Vector{Vector{Float64}})
    # Trouver la position au filet
    clears_net = false
    net_height_at_pass = 0.0

    for i in 2:length(us)
        x1_prev = us[i-1][1]
        x1_curr = us[i][1]
        if x1_prev < X_net && x1_curr >= X_net
            # Interpolation linéaire pour trouver x3 au filet
            α = (X_net - x1_prev) / (x1_curr - x1_prev)
            x3_net = us[i-1][3] + α * (us[i][3] - us[i-1][3])
            net_height_at_pass = x3_net
            clears_net = x3_net >= H_net
            break
        end
    end

    # Trouver le point de rebond (x3 ≤ 0)
    bounce_x1 = NaN
    bounce_x2 = NaN

    for i in 2:length(us)
        if us[i][3] <= 0.0 && us[i-1][3] > 0.0
            # Interpolation
            α = us[i-1][3] / (us[i-1][3] - us[i][3])
            bounce_x1 = us[i-1][1] + α * (us[i][1] - us[i-1][1])
            bounce_x2 = us[i-1][2] + α * (us[i][2] - us[i-1][2])
            break
        end
    end

    in_length  = !isnan(bounce_x1) && (X_net <= bounce_x1 <= 2 * X_net)
    in_width   = !isnan(bounce_x2) && (abs(bounce_x2) <= W_court_half)
    in_court   = clears_net && in_length && in_width

    return (in_court=in_court,
            clears_net=clears_net,
            net_height=net_height_at_pass,
            bounce_x1=bounce_x1,
            bounce_x2=bounce_x2)
end

# ============================================================
# Recherche de la vitesse minimale et maximale (bisection)
# ============================================================

"""
    find_speed_range(shot_type; v_min=10.0, v_max=100.0, tol=0.1)

Trouve par bisection la vitesse minimale et maximale pour que
la balle soit dans le court.
"""
function find_speed_range(shot_type::Symbol;
                          v_min::Float64=5.0,
                          v_max::Float64=100.0,
                          tol::Float64=0.1)

    function valid(speed)
        u0, omega = initial_conditions(speed, shot_type)
        ts, us = simulate(u0, omega; h=0.001, t_max=5.0)
        result = is_on_court(ts, us)
        return result.in_court
    end

    # --- Vitesse minimale (bisection) ---
    lo, hi = v_min, v_max
    # Vérifier qu'il existe une plage valide
    v_valid = NaN
    for v in lo:1.0:hi
        if valid(v)
            v_valid = v
            break
        end
    end
    if isnan(v_valid)
        println("⚠ Aucune vitesse valide trouvée pour $shot_type")
        return (NaN, NaN)
    end

    # Bisection pour v_min
    lo_min = v_min
    hi_min = v_valid
    while hi_min - lo_min > tol
        mid = (lo_min + hi_min) / 2
        if valid(mid)
            hi_min = mid
        else
            lo_min = mid
        end
    end
    v_minimum = (lo_min + hi_min) / 2

    # Bisection pour v_max (cherche où ça devient invalide)
    lo_max = v_valid
    hi_max = v_max
    while hi_max - lo_max > tol
        mid = (lo_max + hi_max) / 2
        if valid(mid)
            lo_max = mid
        else
            hi_max = mid
        end
    end
    v_maximum = (lo_max + hi_max) / 2

    return (v_minimum, v_maximum)
end

# ============================================================
# Affichage des résultats
# ============================================================

function print_trajectory_info(shot_type::Symbol, speed::Float64)
    u0, omega = initial_conditions(speed, shot_type)
    ts, us = simulate(u0, omega; h=0.001, t_max=5.0)
    res = is_on_court(ts, us)

    println("\n--- Coup : $(uppercase(string(shot_type))) | Vitesse : $(round(speed, digits=1)) m/s ---")
    println("  Passe le filet     : $(res.clears_net)  (hauteur = $(round(res.net_height, digits=3)) m, min requis = $H_net m)")
    println("  Rebond x1          : $(round(res.bounce_x1, digits=2)) m  (valide : $(X_net) ≤ x1 ≤ $(2*X_net))")
    println("  Rebond x2          : $(round(res.bounce_x2, digits=2)) m  (valide : |x2| ≤ $W_court_half)")
    println("  Dans le court      : $(res.in_court)")
end

# ============================================================
# MAIN
# ============================================================

println("=" ^ 60)
println("  HAWK-EYE TENNIS — Simulation RK4 (Julia)")
println("=" ^ 60)
println()
println("Paramètres physiques :")
println("  ρ = $ρ kg/m³  |  R = $R m  |  m = $m kg  |  g = $g m/s²")
println("  Terrain : L = $(2*X_net) m  |  l = $(2*W_court_half) m  |  Filet = $H_net m")

# --- Trajectoires exemple à 50 m/s ---
test_speed = 50.0

println("\n$(repeat('─', 60))")
println("Trajectoires à $(test_speed) m/s (angle initial = -10°):")

for shot in [:flat, :topspin, :slice]
    print_trajectory_info(shot, test_speed)
end

# --- Recherche des vitesses min/max ---
println("\n$(repeat('─', 60))")
println("Recherche des vitesses minimale et maximale (bisection) :")
println("(Angle initial = -10°, |ω| = 30 rad/s pour topspin/slice)")
println()

for shot in [:flat, :topspin, :slice]
    print("  $(uppercase(string(shot))) ... ")
    v_lo, v_hi = find_speed_range(shot)
    if isnan(v_lo)
        println("aucune plage valide trouvée.")
    else
        println("v_min ≈ $(round(v_lo, digits=1)) m/s  |  v_max ≈ $(round(v_hi, digits=1)) m/s")
        println("         ($(round(v_lo*3.6, digits=1)) km/h  à  $(round(v_hi*3.6, digits=1)) km/h)")
    end
end

println("\n$(repeat('═', 60))")
println("Simulation terminée.")