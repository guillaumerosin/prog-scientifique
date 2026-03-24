# Exercice Intégration - Chapitre 4 - Rosin Guillaume

# RAPPEL DU COURS : Règle de Simpson
# Sur un intervalle [a,b] avec h = (b-a)/2 :
#   I ≈ (h/3) * (f(x0) + 4f(x1) + f(x2))
#   erreur ε = -(1/90) * h⁵ * f''''(x_max)
#
# PROBLÈME DE RUNGE :
#   Si on applique Simpson une seule fois sur [1,4]
#   avec peu de points → grosse erreur car le polynôme
#   d'interpolation oscille aux bords (Runge's phenomenon)
#
# SOLUTION : Simpson COMPOSITE
#   On divise [1,4] en N sous-intervalles de taille H = (b-a)/N
#   Sur chaque sous-intervalle on applique Simpson (2 points internes)
#   → beaucoup de petits intervalles = polynômes locaux = pas de Runge
# ============================================================

# La fonction à intégrer
f(x) = cos(log(x))    # log() en Julia = ln() naturel

# ============================================================
# Simpson SIMPLE (sur un seul intervalle [a,b])
# Formule du cours : (h/3) * (f(x0) + 4f(x1) + f(x2))
# avec h = (b-a)/2, x0=a, x1=(a+b)/2, x2=b
# ============================================================
function simpson_simple(f, a, b)
    h  = (b - a) / 2.0       # pas = demi-intervalle
    x0 = a
    x1 = (a + b) / 2.0       # point milieu
    x2 = b
    return (h / 3.0) * (f(x0) + 4*f(x1) + f(x2))
end

# ============================================================
# Simpson COMPOSITE (pour éviter Runge)
# On divise [a,b] en N sous-intervalles
# et on applique Simpson simple sur chacun
# ============================================================
function simpson_composite(f, a, b, N)
    H      = (b - a) / N     # taille de chaque sous-intervalle
    total  = 0.0

    for k in 1:N
        # Bornes du k-ième sous-intervalle
        x_left  = a + (k-1) * H
        x_right = a + k * H
        # Ajout de la contribution de Simpson sur ce sous-intervalle
        total += simpson_simple(f, x_left, x_right)
    end

    return total
end

# ============================================================
# Valeur analytique exacte (pour vérifier)
# ∫ cos(ln(x)) dx = (x/2) * (cos(ln(x)) + sin(ln(x))) + C
# ============================================================
function antiderivative(x)
    return (x / 2.0) * (cos(log(x)) + sin(log(x)))
end

exact = antiderivative(4.0) - antiderivative(1.0)

# ============================================================
# RÉSULTATS
# ============================================================
println("=" ^ 55)
println("  ∫₁⁴ cos(ln(x)) dx — Simpson Composite")
println("=" ^ 55)
println()
println("Valeur exacte analytique : $(round(exact, digits=8))")
println()

# On teste avec différents nombres de sous-intervalles
# pour montrer la convergence et l'absence de Runge
println("N sous-intervalles | Résultat Simpson  | Erreur absolue")
println(repeat("-", 55))

for N in [1, 2, 4, 8, 16, 32, 64, 100]
    result = simpson_composite(f, 1.0, 4.0, N)
    err    = abs(result - exact)
    println("  N = $(lpad(N,4))           |  $(round(result, digits=8))  |  $(round(err, sigdigits=3))")
end

println()
println("=" ^ 55)
println("Résultat final avec N=100 sous-intervalles :")
result_final = simpson_composite(f, 1.0, 4.0, 100)
println("  ∫₁⁴ cos(ln(x)) dx ≈ $(round(result_final, digits=8))")
println("  Erreur              ≈ $(round(abs(result_final - exact), sigdigits=3))")
println("=" ^ 55)