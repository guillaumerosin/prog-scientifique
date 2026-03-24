# Constantes physiques 
const ρ = 1.23          # kg/m³  — densité air
const R = 0.0335        # m      — rayon balle
const m = 0.058         # kg     — masse balle
const g = 9.81          # m/s²   — gravité

# Fonction MD : calcule M et D
function MD(q, x, omega, v)

    # --- Coefficients aérodynamiques 
    if isinf(q)
        # Cas flat 
        CM = 0.0
        CD = 0.508
    else
        CM = 1.0 / (2.022 + 0.981 * q)
        CD = 0.508 + (1.0 / (22.053 + 4.196 * q^(5/2)))^(2/5)
    end

    v_norm = norm(v)

    # Traînée D = -0.5 * ρ * CD * π * R² * |v|² * (v/|v|) ---
    D = -0.5 * ρ * CD * π * R^2 * v_norm .* v ./ v_norm

    # --- Magnus M = 0.5 * ρ * CM * π * R² * |v|² * (ω/|ω|) × (v/|v|) ---
    omega_norm = norm(omega)
    if omega_norm < 1e-10 || v_norm < 1e-10
        M = [0.0, 0.0, 0.0]
    else
        omega_hat = omega ./ omega_norm    # ω/|ω|
        v_hat     = v ./ v_norm            # v/|v|
        # produit vectoriel 
        cross_vec = [
            omega_hat[2]*v_hat[3] - omega_hat[3]*v_hat[2],
            omega_hat[3]*v_hat[1] - omega_hat[1]*v_hat[3],
            omega_hat[1]*v_hat[2] - omega_hat[2]*v_hat[1]
        ]
        M = 0.5 * ρ * CM * π * R^2 * v_norm^2 .* cross_vec
    end

    return (M=M, D=D)
end
# Fonction tennis(omega) 
# Retourne la fonction f(t,ut) pour le rk4
function tennis(omega)

    X1, X2, X3, dX1, dX2, dX3 = 1, 2, 3, 4, 5, 6

    f(t, ut) =
        let
            v = [ ut[dX1], ut[dX2], ut[dX3] ]
            x = [ ut[X1],  ut[X2],  ut[X3]  ]

            omega_norm = norm(omega)
            q = omega_norm < 1e-10 ? Inf : norm(v) / (R * omega_norm)

            mdValues = MD(q, x, omega, v)

            [
                ut[dX1],                                          # dx1/dt = v1
                ut[dX2],                                          # dx2/dt = v2
                ut[dX3],                                          # dx3/dt = v3
                (mdValues.M[X1] + mdValues.D[X1]) ./ m,          # d²x1 = (M1+D1)/m
                (mdValues.M[X2] + mdValues.D[X2]) ./ m,          # d²x2 = (M2+D2)/m
                -g + (mdValues.M[X3] + mdValues.D[X3]) ./ m      # d²x3 = -g + (M3+D3)/m
            ]
        end

    return f
end


# RK4 
# rk4(f, u0, a, b, n)
#   f  : fonction f(t, ut)
#   u0 : condition initiale
#   a  : temps début
#   b  : temps fin
#   n  : nombre de points
function rk4(f, u0, a, b, n)

    h = (b - a) / n                       
    u = [u0 for i in a:b]                 
    u = Vector{Vector{Float64}}(undef, n)
    u[1] = u0

    for i in 2:n                           

        t_current = i * h + a              

        k1 = f(t_current,       u[i-1])              
        k2 = f(t_current + h/2, u[i-1] + h * k1/2)  
        k3 = f(t_current + h/2, u[i-1] + h * k2/2)  
        k4 = f(t_current + h,   u[i-1] + h * k3)     

        u[i] = u[i-1] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

       
        if u[i][3] <= 0.0
            return u[1:i]
        end
    end

    return u
end

norm(v) = sqrt(sum(v.^2))


v_init = 25.0        #

# FLAT 
omega = [0.0, 0.0, 0.0]    


u0 = [-11.89, 0.0, 0.75,
       v_init * cos(π/18), 0.0, v_init * sin(π/18)]

flat = rk4(tennis(omega), u0, 0.0, 2.0, 2000)

# --- TOPSPIN ---
omega_top = [0.0, 30.0, 0.0]    # ωi > 0 (slide : topspin)
topspin   = rk4(tennis(omega_top), u0, 0.0, 2.0, 2000)

# --- SLICE ---
omega_sli = [0.0, -30.0, 0.0]   # ωi < 0 (slide : slice)
slice     = rk4(tennis(omega_sli), u0, 0.0, 2.0, 2000)

println("=== Hawk-Eye Tennis — Résultats ===\n")

for (nom, traj) in [("FLAT", flat), ("TOPSPIN", topspin), ("SLICE", slice)]
    final = traj[end]
    x1_final = final[1] + 11.89   
    x3_final = final[3]

    println("$nom :")
    println("  Rebond en x1 = $(round(x1_final, digits=2)) m  (valide : 11.89 à 23.78 m)")
    println("  Hauteur finale x3 = $(round(x3_final, digits=3)) m")
    println("  Nombre de pas calculés : $(length(traj))")
    println()
end