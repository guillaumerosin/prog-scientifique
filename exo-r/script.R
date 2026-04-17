# =============================================================================
# Exercice - Comparaison d'algorithmes
# Scientific Programming - Chapter 5
# Variable aléatoire X = k, où time(algo(input)) = a × input^k
# =============================================================================
# 
library(tidyverse)

# =============================================================================
# 1. CHARGEMENT ET EXPLORATION DES DONNÉES
# =============================================================================

algos <- read_csv("algos.csv")

# cat("=== Aperçu des données ===\n")
print(head(algos))
cat("\n=== Résumé statistique ===\n")
print(summary(algos))


# =============================================================================
# 2. VISUALISATION — Scatter plots (indice 1)
# =============================================================================

par(mfrow = c(1, 3))

plot(algos$Cremer,
     main = "Scatter — Cremer",
     ylab = "k", xlab = "Observation",
     col = "darkblue", pch = 16)

plot(algos$Quevy,
     main = "Scatter — Quevy",
     ylab = "k", xlab = "Observation",
     col = "darkred", pch = 16)

plot(algos$Lerat,
     main = "Scatter — Lerat",
     ylab = "k", xlab = "Observation",
     col = "darkgreen", pch = 16)

par(mfrow = c(1, 1))


# =============================================================================
# 3. VISUALISATION — Histogrammes (indice 2)
# =============================================================================

par(mfrow = c(1, 3))

hist(algos$Cremer,
     main = "Histogramme — Cremer",
     xlab = "k", col = "lightblue",
     probability = TRUE)
curve(dnorm(x, mean = mean(algos$Cremer), sd = sd(algos$Cremer)),
      add = TRUE, col = "darkblue", lwd = 2)

hist(algos$Quevy,
     main = "Histogramme — Quevy",
     xlab = "k", col = "lightcoral",
     probability = TRUE)
curve(dnorm(x, mean = mean(algos$Quevy), sd = sd(algos$Quevy)),
      add = TRUE, col = "darkred", lwd = 2)

hist(algos$Lerat,
     main = "Histogramme — Lerat",
     xlab = "k", col = "lightgreen",
     probability = TRUE)
curve(dnorm(x, mean = mean(algos$Lerat), sd = sd(algos$Lerat)),
      add = TRUE, col = "darkgreen", lwd = 2)

par(mfrow = c(1, 1))


# =============================================================================
# 4. ESTIMATION DES PARAMÈTRES (indice 3)
# =============================================================================

mu_C  <- mean(algos$Cremer) ; sd_C  <- sd(algos$Cremer) ; var_C <- var(algos$Cremer)
mu_Q  <- mean(algos$Quevy)  ; sd_Q  <- sd(algos$Quevy)  ; var_Q <- var(algos$Quevy)
mu_L  <- mean(algos$Lerat)  ; sd_L  <- sd(algos$Lerat)  ; var_L <- var(algos$Lerat)

cat("\n=== Paramètres estimés N(µ, σ²) ===\n")
cat(sprintf("Cremer : µ = %.4f | σ = %.4f | σ² = %.4f\n", mu_C, sd_C, var_C))
cat(sprintf("Quevy  : µ = %.4f | σ = %.4f | σ² = %.4f\n", mu_Q, sd_Q, var_Q))
cat(sprintf("Lerat  : µ = %.4f | σ = %.4f | σ² = %.4f\n", mu_L, sd_L, var_L))


# =============================================================================
# 5. TEST D'ADÉQUATION — Kolmogorov-Smirnov (indice 4)
# H0 : les données suivent N(µ, σ²)
# p > 0.05 → on ne rejette pas H0 → distribution normale acceptable
# =============================================================================

cat("\n=== Test d'adéquation — Kolmogorov-Smirnov ===\n")

ks_C <- ks.test(algos$Cremer, "pnorm", mean = mu_C, sd = sd_C)
ks_Q <- ks.test(algos$Quevy,  "pnorm", mean = mu_Q, sd = sd_Q)
ks_L <- ks.test(algos$Lerat,  "pnorm", mean = mu_L, sd = sd_L)

cat(sprintf("Cremer : D = %.4f | p-value = %.4f | Normal ? %s\n",
            ks_C$statistic, ks_C$p.value,
            ifelse(ks_C$p.value > 0.05, "OUI (p > 0.05)", "NON (p <= 0.05)")))

cat(sprintf("Quevy  : D = %.4f | p-value = %.4f | Normal ? %s\n",
            ks_Q$statistic, ks_Q$p.value,
            ifelse(ks_Q$p.value > 0.05, "OUI (p > 0.05)", "NON (p <= 0.05)")))

cat(sprintf("Lerat  : D = %.4f | p-value = %.4f | Normal ? %s\n",
            ks_L$statistic, ks_L$p.value,
            ifelse(ks_L$p.value > 0.05, "OUI (p > 0.05)", "NON (p <= 0.05)")))


# =============================================================================
# 6. TEST DE CONFORMITÉ — F-test Fisher-Snedecor (variances)
# H0 : σ_A = σ_B
# p > 0.05 → variances égales → on peut utiliser var.equal = TRUE dans t.test
# =============================================================================

cat("\n=== Test de conformité — F-test (variances) ===\n")

ftest_CQ <- var.test(algos$Cremer, algos$Quevy)
ftest_CL <- var.test(algos$Cremer, algos$Lerat)
ftest_QL <- var.test(algos$Quevy,  algos$Lerat)

cat(sprintf("Cremer vs Quevy : F = %.4f | p-value = %.4f | Variances égales ? %s\n",
            ftest_CQ$statistic, ftest_CQ$p.value,
            ifelse(ftest_CQ$p.value > 0.05, "OUI", "NON")))

cat(sprintf("Cremer vs Lerat : F = %.4f | p-value = %.4f | Variances égales ? %s\n",
            ftest_CL$statistic, ftest_CL$p.value,
            ifelse(ftest_CL$p.value > 0.05, "OUI", "NON")))

cat(sprintf("Quevy  vs Lerat : F = %.4f | p-value = %.4f | Variances égales ? %s\n",
            ftest_QL$statistic, ftest_QL$p.value,
            ifelse(ftest_QL$p.value > 0.05, "OUI", "NON")))


# =============================================================================
# 7. TEST DE CONFORMITÉ — Student t-test (moyennes)
# H0 : µ_A = µ_B
# p <= 0.05 → moyennes significativement différentes → un algo est meilleur
# var.equal : adapter selon les résultats du F-test ci-dessus
# =============================================================================

cat("\n=== Test de conformité — t-test Student (moyennes) ===\n")

# var.equal = TRUE si le F-test correspondant a p > 0.05
ttest_CQ <- t.test(algos$Cremer, algos$Quevy,
                   var.equal = (ftest_CQ$p.value > 0.05))
ttest_CL <- t.test(algos$Cremer, algos$Lerat,
                   var.equal = (ftest_CL$p.value > 0.05))
ttest_QL <- t.test(algos$Quevy,  algos$Lerat,
                   var.equal = (ftest_QL$p.value > 0.05))

cat(sprintf("Cremer vs Quevy : t = %.4f | p-value = %.4f | Moyennes différentes ? %s\n",
            ttest_CQ$statistic, ttest_CQ$p.value,
            ifelse(ttest_CQ$p.value <= 0.05, "OUI (p <= 0.05)", "NON (p > 0.05)")))

cat(sprintf("Cremer vs Lerat : t = %.4f | p-value = %.4f | Moyennes différentes ? %s\n",
            ttest_CL$statistic, ttest_CL$p.value,
            ifelse(ttest_CL$p.value <= 0.05, "OUI (p <= 0.05)", "NON (p > 0.05)")))

cat(sprintf("Quevy  vs Lerat : t = %.4f | p-value = %.4f | Moyennes différentes ? %s\n",
            ttest_QL$statistic, ttest_QL$p.value,
            ifelse(ttest_QL$p.value <= 0.05, "OUI (p <= 0.05)", "NON (p > 0.05)")))


# =============================================================================
# 8. RÉPONSES AUX QUESTIONS
# =============================================================================

cat("\n=== RÉPONSES AUX QUESTIONS ===\n")

# Q1 — Un algorithme est-il meilleur qu'un autre ?
cat("\nQ1 — Un algorithme est-il meilleur qu'un autre ?\n")
cat("  → Comparer les p-values des t-tests :\n")
cat(sprintf("     Cremer vs Quevy : %s\n",
            ifelse(ttest_CQ$p.value <= 0.05,
                   "DIFFERENT — un algo est significativement meilleur",
                   "PAS de différence significative")))
cat(sprintf("     Cremer vs Lerat : %s\n",
            ifelse(ttest_CL$p.value <= 0.05,
                   "DIFFERENT — un algo est significativement meilleur",
                   "PAS de différence significative")))
cat(sprintf("     Quevy  vs Lerat : %s\n",
            ifelse(ttest_QL$p.value <= 0.05,
                   "DIFFERENT — un algo est significativement meilleur",
                   "PAS de différence significative")))

# Q2 — Lequel utiliser (en général) ?
moyennes <- c(Cremer = mu_C, Quevy = mu_Q, Lerat = mu_L)
meilleur <- names(which.min(moyennes))

cat("\nQ2 — Lequel utiliser ?\n")
cat(sprintf("  → Algorithme avec la plus petite moyenne µ_k : %s (µ = %.4f)\n",
            meilleur, min(moyennes)))
cat("  → Un k plus petit = complexité plus faible = algorithme plus rapide\n")

# Q3 — Lequel sur un nombre croissant d'entrées ?
cat("\nQ3 — Lequel sur un nombre croissant d'entrées ?\n")
cat(sprintf("  → time = a × input^k → quand input → ∞, c'est k qui domine\n"))
cat(sprintf("  → Le plus petit µ_k donne la croissance la plus lente\n"))
cat(sprintf("  → Réponse : %s (µ_k = %.4f)\n", meilleur, min(moyennes)))
cat("  → Note : si σ diffère, préférer aussi le moins variable sur de grands datasets\n")