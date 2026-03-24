#############################
# Exemple simple de problème de Cauchy
# u'(t) = f(t, u) = u
# u(0) = 1
# t ∈ [0, 2]
#############################

# On définit d'abord la fonction f(t, u) du problème :
# u'(t) = f(t, u(t))
function f(t, u)
    # Ici, l'équation différentielle est u'(t) = u(t)
    return u
end

########################################
# Méthode d'Euler explicite pour un IVP
########################################

"""
    euler(f, t0, u0, h, n)

Résout numériquement le problème de Cauchy

    u'(t) = f(t, u(t))
    u(t0) = u0

à l'aide de la méthode d'Euler explicite.

Arguments :
- `f`  : fonction f(t, u) définissant l'ODE
- `t0` : temps initial
- `u0` : valeur initiale u(t0)
- `h`  : pas de temps (taille de l'incrément en t)
- `n`  : nombre de pas à effectuer

Retourne :
- un vecteur `ts` contenant les temps
- un vecteur `us` contenant les approximations de u(t)
"""
function euler(f, t0, u0, h, n)
    # Allocation des vecteurs de résultats
    # ts : les temps (t0, t1, t2, ...)
    # us : les valeurs u(t0), u(t1), ...
    ts = zeros(Float64, n+1)
    us = zeros(Float64, n+1)

    # Initialisation avec la condition initiale
    ts[1] = t0      # t_0
    us[1] = u0      # u(t_0) = u0

    # Boucle principale de la méthode d'Euler
    # Pour k = 0, 1, ..., n-1 :
    #   t_{k+1} = t_k + h
    #   u_{k+1} = u_k + h * f(t_k, u_k)
    for k in 1:n
        # Temps suivant
        ts[k+1] = ts[k] + h

        # Pente locale donnée par l'ODE : u'(t_k) = f(t_k, u_k)
        slope = f(ts[k], us[k])

        # Mise à jour de la solution : schéma d'Euler explicite
        us[k+1] = us[k] + h * slope
    end

    return ts, us
end

###################################
# Utilisation de la méthode d'Euler
###################################

# Paramètres du problème
t0 = 0.0       # temps initial
u0 = 1.0       # condition initiale u(0) = 1
t_end = 2.0    # temps final
h = 0.1        # pas de temps (plus petit => plus précis mais plus coûteux)

# Nombre de pas : on va de t0 à t_end par pas de taille h
n = Int((t_end - t0) / h)

# Appel de la méthode d'Euler
ts, us = euler(f, t0, u0, h, n)

##########################################
# Comparaison avec la solution exacte e^t
##########################################

# On définit la solution exacte pour ce problème
exact_solution(t) = exp(t)

println("t       u_Euler(t)      u_exacte(t)       erreur |u_Euler - u_exacte|")
println("-------------------------------------------------------------------")
for i in 1:length(ts)
    t = ts[i]
    u_approx = us[i]
    u_exact = exact_solution(t)
    err = abs(u_approx - u_exact)
    @printf("%.2f    %12.8f    %12.8f    %12.8f\n", t, u_approx, u_exact, err)
end