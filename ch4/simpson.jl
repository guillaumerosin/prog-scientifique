# Intégration numérique par la règle de Simpson composite
# Calcul de ∫₁⁴ cos(ln(x)) dx

# Fonction à intégrer
f(x) = cos(log(x))

# Règle de Simpson composite
function simpson_composite(f, a, b, N)
    if N % 2 != 0
        error("N doit être pair pour la règle de Simpson.")
    end

    h = (b - a) / N
    somme = f(a) + f(b)

    for i in 1:(N-1)
        xi = a + i * h
        coeff = (i % 2 == 0) ? 2 : 4
        somme += coeff * f(xi)
    end

    return (h / 3) * somme
end

# Paramètres
a = 1.0
b = 4.0

println("=" ^ 50)
println("Intégrale de cos(ln(x)) sur [1, 4]")
println("Règle de Simpson composite")
println("=" ^ 50)
println()

# Affichage pour différentes valeurs de N
println("  N  |   Résultat     | Erreur relative")
println("-" ^ 42)

exact = simpson_composite(f, a, b, 10000)  # valeur de référence

for N in [2, 4, 6, 8, 10, 20, 50, 100]
    result = simpson_composite(f, a, b, N)
    err = abs((result - exact) / exact) * 100
    @printf("  %3d | %14.8f | %.6f %%\n", N, result, err)
end

println()
println("Résultat avec N = 4 (comme dans l'exercice) :")
N = 4
h = (b - a) / N
result = simpson_composite(f, a, b, N)

println()
println("Détail des points :")
println("  i  |    xᵢ    |  f(xᵢ)   | Coeff | Coeff × f(xᵢ)")
println("-" ^ 60)
for i in 0:N
    xi = a + i * h
    yi = f(xi)
    coeff = (i == 0 || i == N) ? 1 : (i % 2 == 0 ? 2 : 4)
    @printf("  %2d | %8.4f | %8.6f |   %d   | %8.6f\n", i, xi, yi, coeff, coeff * yi)
end

println()
@printf("h = (b - a) / N = (%.1f - %.1f) / %d = %.4f\n", b, a, N, h)
@printf("I ≈ (h/3) × Σ = %.6f\n", result)