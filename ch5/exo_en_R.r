# =============================================================================
# EXERCICE - Chapitre 6 : R et Tests Statistiques
# Auteur : Exercice basé sur le cours de J-S Lerat (HEH Technologies)
# =============================================================================
#
# ============================================================
# POURQUOI CE COURS ET CET EXERCICE ?
# ============================================================
#
# SLIDE 1-5 : Introduction à R / RStudio
# ----------------------------------------
# R est un langage spécialement conçu pour le traitement statistique.
# Contrairement à Python ou Julia, R a été pensé dès le départ pour
# manipuler des données et faire des tests de probabilités.
# RStudio est l'IDE (environnement de développement) recommandé pour R.
# On l'utilise ici car l'exercice demande d'analyser des données
# d'algorithmes avec des outils statistiques.
#
# SLIDE 6-9 : Types, Variables, Opérateurs
# ------------------------------------------
# R utilise <- pour l'affectation (et non = comme en Python/Java).
# Les vecteurs (c(...)) sont la structure de base de R : tout est vecteur.
# Les opérateurs arithmétiques (+, -, *, /, ^, %%) fonctionnent
# directement sur des vecteurs (element-wise), ce qui est très pratique
# pour traiter des colonnes de données sans boucles.
#
# SLIDE 10-11 : Décrire les données (Describing Data)
# -----------------------------------------------------
# Avant tout test, on "décrit" les données pour en comprendre la forme :
#   - summary() : vue d'ensemble (min, max, médiane, moyenne, quartiles)
#   - median()   : valeur centrale (robuste aux valeurs extrêmes)
#   - mean()     : moyenne arithmétique (sensible aux valeurs extrêmes)
#   - sd() / var() : écart-type et variance (mesure la dispersion)
#   - cor()      : corrélation entre variables
# Dans l'exercice, on décrit X = k (l'exposant de complexité) pour
# chaque algorithme afin de comprendre leur comportement.
#
# SLIDE 12-18 : Distributions
# ----------------------------
# Une distribution décrit comment les valeurs d'une variable se répartissent.
# On en distingue deux familles :
#   - CONTINUES : Uniforme, Normale (Gaussienne), Gamma, Exponentielle, Beta
#   - DISCRÈTES  : Binomiale, Poisson, Bernoulli
#
# Dans R, chaque distribution a 4 fonctions préfixées :
#   d<dist>() = densité (PDF)       | ex: dnorm(), dunif(), dbinom()
#   p<dist>() = probabilité cumulée | ex: pnorm()
#   q<dist>() = quantile            | ex: qnorm()
#   r<dist>() = génération aléatoire| ex: rnorm()
#
# La distribution NORMALE est cruciale pour cet exercice :
# x ~ N(µ, σ) avec :
#   µ = moyenne (position du pic)
#   σ = écart-type (largeur du pic)
#   σ² = variance
# Si nos valeurs de k suivent une loi normale, on peut utiliser des
# tests paramétriques puissants (t-test, F-test).
#
# SLIDE 18 : Visualisation
# -------------------------
# hist() crée un histogramme qui permet de "voir" visuellement si une
# distribution ressemble à une gaussienne, une uniforme, etc.
# C'est la PREMIÈRE étape du diagnostic avant tout test formel.
#
# SLIDE 19-22 : Tests d'hypothèses - Adéquation
# -----------------------------------------------
# Un test d'ADÉQUATION vérifie si un échantillon suit une loi donnée.
# H0 = "les données suivent cette loi" (hypothèse nulle à tester)
# H1 = "les données ne suivent PAS cette loi" (hypothèse alternative)
#
# Règle universelle : si p-value <= 0.05 → rejeter H0
#                     si p-value > 0.05  → ne pas rejeter H0
#
# Tests utilisés :
#   ks.test()    : Kolmogorov-Smirnov (pour distributions CONTINUES)
#                  Compare la distribution empirique à une loi théorique.
#                  Mesure le max écart entre la CDF empirique et théorique.
#
#   chisq.test() : Chi² de Pearson (pour distributions DISCRÈTES ou catégories)
#                  Compare les fréquences observées aux fréquences théoriques.
#
# Dans l'exercice : on teste si X=k suit une loi normale pour chaque algo.
#
# SLIDE 23-24 : Tests de conformité
# -----------------------------------
# Un test de CONFORMITÉ compare deux échantillons qui suivent (supposément)
# la même loi, pour vérifier si leurs PARAMÈTRES sont identiques.
#
#   var.test()  : Test F de Fisher-Snedecor
#                 H0 = "les deux variances sont égales" (σA = σB)
#                 Prérequis avant le t-test !
#
#   t.test()    : Test t de Student
#                 H0 = "les deux moyennes sont égales" (µA = µB)
#                 Si µ différentes → les algos ont des complexités différentes
#                 Si var.test non significatif → var.equal = TRUE dans t.test
#
# Dans l'exercice : on compare les k des 3 algorithmes deux à deux.
# Si les moyennes diffèrent → un algo est structurellement plus rapide.
#
# ============================================================
# FIN DE L'EXPLICATION PÉDAGOGIQUE - DÉBUT DU CODE
# ============================================================


# -----------------------------------------------------------------------------
# 0. CHARGEMENT DES LIBRAIRIES
# -----------------------------------------------------------------------------
# tidyverse inclut ggplot2 (visualisation), dplyr (manipulation), readr (CSV)
# Si non installé : install.packages("tidyverse")
library(tidyverse)


# -----------------------------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES
# (Slide "Describing Data" - Load a Dataframe)
# -----------------------------------------------------------------------------
# Le fichier algos.csv contient les mesures de k pour chaque algorithme.
# On suppose qu'il est dans le répertoire de travail courant.
# Adapter le chemin si nécessaire.

algos <- read_csv("algos.csv")

# Aperçu rapide de la structure
cat("=== STRUCTURE DU DATAFRAME ===\n")
print(str(algos))
cat("\n")

# summary() donne min, Q1, médiane, moyenne, Q3, max pour chaque colonne
# C'est la première description statistique (Slide 11)
cat("=== RÉSUMÉ STATISTIQUE ===\n")
print(summary(algos))
cat("\n")


# -----------------------------------------------------------------------------
# 2. EXTRACTION DES DONNÉES PAR ALGORITHME
# -----------------------------------------------------------------------------
# On suppose que le CSV a 3 colonnes : Cremer, Quevy, Lerat
# (ou similaire - adapter les noms de colonnes selon le vrai CSV)

k_cremer <- algos$Cremer
k_quevy  <- algos$Quevy
k_lerat  <- algos$Lerat


# -----------------------------------------------------------------------------
# 3. DESCRIPTION DES DONNÉES (Slide 11 - Describing Data)
# -----------------------------------------------------------------------------
# Calcul des paramètres descriptifs pour chaque algorithme

cat("=== PARAMÈTRES DESCRIPTIFS ===\n")

cat("\n--- Algorithme CREMER ---\n")
cat("Médiane :", median(k_cremer), "\n")
cat("Moyenne :", mean(k_cremer), "\n")
cat("Écart-type :", sd(k_cremer), "\n")
cat("Variance :", var(k_cremer), "\n")

cat("\n--- Algorithme QUEVY ---\n")
cat("Médiane :", median(k_quevy), "\n")
cat("Moyenne :", mean(k_quevy), "\n")
cat("Écart-type :", sd(k_quevy), "\n")
cat("Variance :", var(k_quevy), "\n")

cat("\n--- Algorithme LERAT ---\n")
cat("Médiane :", median(k_lerat), "\n")
cat("Moyenne :", mean(k_lerat), "\n")
cat("Écart-type :", sd(k_lerat), "\n")
cat("Variance :", var(k_lerat), "\n")
cat("\n")

# INTERPRÉTATION :
# Une moyenne/médiane plus FAIBLE = exposant k plus petit = algo plus rapide
# car time = a * input^k : plus k est petit, plus le temps croît lentement


# -----------------------------------------------------------------------------
# 4. VISUALISATION - HINT 1 : Scatter plot (nuage de points)
# -----------------------------------------------------------------------------
# But : voir si les valeurs de k sont stables ou varient beaucoup
# Un bon algo aura des k groupés autour d'une valeur faible

cat("=== VISUALISATION - SCATTER PLOTS ===\n")

# Mise en forme longue pour ggplot2
algos_long <- algos %>%
  mutate(mesure = row_number()) %>%
  pivot_longer(cols = -mesure, names_to = "algorithme", values_to = "k")

# Scatter plot
p1 <- ggplot(algos_long, aes(x = mesure, y = k, color = algorithme)) +
  geom_point(alpha = 0.6, size = 2) +
  labs(
    title = "Scatter plot des valeurs de k par algorithme",
    subtitle = "time(algo(input)) = a × input^k   →   X = k",
    x = "Numéro de mesure",
    y = "Valeur de k",
    color = "Algorithme"
  ) +
  theme_minimal() +
  facet_wrap(~algorithme, ncol = 1)

print(p1)
ggsave("scatter_k.png", p1, width = 8, height = 6)
cat("→ Fichier 'scatter_k.png' sauvegardé\n\n")

# OBSERVATION ATTENDUE :
# Si les points sont bien groupés → distribution stable → algo prévisible
# Si les points sont dispersés → variabilité élevée → comportement imprévisible


# -----------------------------------------------------------------------------
# 5. VISUALISATION - HINT 2 : Histogrammes
# (Slide 18 - Vizualisation avec hist())
# -----------------------------------------------------------------------------
# But : observer si la distribution de k ressemble à une cloche (gaussienne)
# C'est un pré-test VISUEL avant les tests formels d'adéquation

cat("=== VISUALISATION - HISTOGRAMMES ===\n")

p2 <- ggplot(algos_long, aes(x = k, fill = algorithme)) +
  geom_histogram(bins = 15, color = "white", alpha = 0.8) +
  labs(
    title = "Histogramme de k par algorithme",
    subtitle = "Forme en cloche = distribution normale probable",
    x = "Valeur de k",
    y = "Fréquence"
  ) +
  theme_minimal() +
  facet_wrap(~algorithme, ncol = 1) +
  theme(legend.position = "none")

print(p2)
ggsave("histogrammes_k.png", p2, width = 8, height = 7)
cat("→ Fichier 'histogrammes_k.png' sauvegardé\n\n")

# OBSERVATION ATTENDUE :
# Si l'histogramme ressemble à une courbe en cloche symétrique
# → on peut supposer une distribution normale
# → les tests paramétriques (KS, t-test, F-test) sont applicables


# -----------------------------------------------------------------------------
# 6. HINT 3 : Estimation des paramètres de la loi normale
# (Slide 15 - Distribution Normale : x ~ N(µ, σ))
# -----------------------------------------------------------------------------
# Pour une loi normale, les estimateurs naturels sont :
#   µ̂ = mean(x)   (estimateur de la moyenne)
#   σ̂ = sd(x)     (estimateur de l'écart-type)

cat("=== ESTIMATION DES PARAMÈTRES N(µ, σ) ===\n")

mu_cremer <- mean(k_cremer)
sd_cremer <- sd(k_cremer)

mu_quevy  <- mean(k_quevy)
sd_quevy  <- sd(k_quevy)

mu_lerat  <- mean(k_lerat)
sd_lerat  <- sd(k_lerat)

cat(sprintf("Cremer : µ = %.4f  |  σ = %.4f\n", mu_cremer, sd_cremer))
cat(sprintf("Quevy  : µ = %.4f  |  σ = %.4f\n", mu_quevy,  sd_quevy))
cat(sprintf("Lerat  : µ = %.4f  |  σ = %.4f\n", mu_lerat,  sd_lerat))
cat("\n")

# Visualisation : superposition histogramme + courbe normale ajustée
p3 <- ggplot(algos_long, aes(x = k)) +
  geom_histogram(aes(y = after_stat(density)), bins = 15,
                 fill = "steelblue", color = "white", alpha = 0.7) +
  stat_function(
    fun = dnorm,
    args = list(mean = mean(algos_long$k), sd = sd(algos_long$k)),
    color = "red", linewidth = 1.2, linetype = "dashed"
  ) +
  labs(
    title = "Distribution de k avec courbe normale ajustée",
    subtitle = "Courbe rouge = N(µ̂, σ̂) estimée   |   Barres = fréquences observées",
    x = "k", y = "Densité"
  ) +
  theme_minimal() +
  facet_wrap(~algorithme, ncol = 1)

print(p3)
ggsave("normal_fit_k.png", p3, width = 8, height = 7)
cat("→ Fichier 'normal_fit_k.png' sauvegardé\n\n")


# -----------------------------------------------------------------------------
# 7. HINT 4 : TEST D'ADÉQUATION - Kolmogorov-Smirnov
# (Slide 21 - Adequacy : Kolmogorov-Smirnov pour distributions continues)
# -----------------------------------------------------------------------------
# H0 : les données suivent une loi normale N(µ̂, σ̂)
# H1 : les données NE suivent PAS une loi normale
#
# Règle : p-value <= 0.05 → rejeter H0 (pas normal)
#         p-value >  0.05 → ne pas rejeter H0 (compatible avec normal)
#
# ATTENTION : ks.test() teste contre une distribution COMPLÈTEMENT spécifiée.
# On utilise les paramètres estimés µ̂ et σ̂ de l'échantillon lui-même.

cat("=== TEST D'ADÉQUATION : Kolmogorov-Smirnov ===\n")
cat("H0 : les données suivent une loi normale N(µ̂, σ̂)\n")
cat("Règle de décision : p-value <= 0.05 → rejeter H0\n\n")

ks_cremer <- ks.test(k_cremer, "pnorm", mean = mu_cremer, sd = sd_cremer)
ks_quevy  <- ks.test(k_quevy,  "pnorm", mean = mu_quevy,  sd = sd_quevy)
ks_lerat  <- ks.test(k_lerat,  "pnorm", mean = mu_lerat,  sd = sd_lerat)

cat("--- KS Test CREMER ---\n")
print(ks_cremer)
cat(ifelse(ks_cremer$p.value > 0.05,
           "→ p > 0.05 : On ne rejette pas H0, Cremer semble normal\n\n",
           "→ p <= 0.05 : On rejette H0, Cremer ne semble PAS normal\n\n"))

cat("--- KS Test QUEVY ---\n")
print(ks_quevy)
cat(ifelse(ks_quevy$p.value > 0.05,
           "→ p > 0.05 : On ne rejette pas H0, Quevy semble normal\n\n",
           "→ p <= 0.05 : On rejette H0, Quevy ne semble PAS normal\n\n"))

cat("--- KS Test LERAT ---\n")
print(ks_lerat)
cat(ifelse(ks_lerat$p.value > 0.05,
           "→ p > 0.05 : On ne rejette pas H0, Lerat semble normal\n\n",
           "→ p <= 0.05 : On rejette H0, Lerat ne semble PAS normal\n\n"))


# -----------------------------------------------------------------------------
# 8. HINT 5 : TEST DE CONFORMITÉ - Comparer les algorithmes
# (Slides 23-24 - Conformity : F-test puis t-test)
# -----------------------------------------------------------------------------
# On compare les algorithmes deux à deux.
# ÉTAPE A : var.test() → Fisher F-test pour l'égalité des variances
#   H0 : σ_A = σ_B  (variances identiques)
#   → Résultat conditionne le paramètre var.equal du t.test()
#
# ÉTAPE B : t.test() → Student pour l'égalité des moyennes
#   H0 : µ_A = µ_B  (moyennes identiques = algos statistiquement équivalents)
#   → Si p-value > 0.05 : les algos ont la même complexité moyenne
#   → Si p-value <= 0.05 : les algos ont des complexités DIFFÉRENTES

comparaisons <- list(
  list(nom = "Cremer vs Quevy", a = k_cremer, b = k_quevy),
  list(nom = "Cremer vs Lerat", a = k_cremer, b = k_lerat),
  list(nom = "Quevy vs Lerat",  a = k_quevy,  b = k_lerat)
)

cat("=== TESTS DE CONFORMITÉ ===\n\n")

for (comp in comparaisons) {
  cat(paste0("--- ", comp$nom, " ---\n"))

  # Étape A : Test F sur les variances
  ftest <- var.test(comp$a, comp$b)
  variances_egales <- ftest$p.value > 0.05
  cat(sprintf("F-test (variances) : p-value = %.4f  →  %s\n",
              ftest$p.value,
              ifelse(variances_egales,
                     "variances ÉGALES (H0 non rejetée)",
                     "variances DIFFÉRENTES (H0 rejetée)")))

  # Étape B : Test t sur les moyennes
  # var.equal = TRUE si le F-test n'a pas rejeté H0
  ttest <- t.test(comp$a, comp$b, var.equal = variances_egales)
  cat(sprintf("t-test (moyennes)  : p-value = %.4f  →  %s\n",
              ttest$p.value,
              ifelse(ttest$p.value > 0.05,
                     "moyennes ÉGALES → algos statistiquement identiques",
                     "moyennes DIFFÉRENTES → algos statistiquement distincts")))
  cat("\n")
}


# -----------------------------------------------------------------------------
# 9. RÉPONSES AUX QUESTIONS DE L'EXERCICE
# -----------------------------------------------------------------------------

cat("=============================================================\n")
cat("RÉPONSES AUX QUESTIONS\n")
cat("=============================================================\n\n")

cat("QUESTION 1 : Un algorithme est-il meilleur qu'un autre ?\n")
cat("  → Voir les résultats des t-tests ci-dessus.\n")
cat("  → Si p-value <= 0.05 entre deux algos : ils sont DIFFÉRENTS.\n")
cat("  → L'algo avec la PLUS PETITE MOYENNE de k est le plus efficace\n")
cat("    car time = a × input^k croît moins vite avec un k faible.\n\n")

cat("QUESTION 2 : Lequel utiliser en général ?\n")
cat("  → Celui avec la plus petite moyenne de k (voir section 3).\n")
cat("  → Mais aussi celui avec le plus faible écart-type (plus prévisible).\n\n")

cat("QUESTION 3 : Lequel utiliser pour un nombre croissant d'inputs ?\n")
cat("  → ABSOLUMENT celui avec le plus petit k moyen.\n")
cat("  → Car pour de grandes valeurs d'input, input^k explose exponentiellement.\n")
cat("  → Exemple : input=1000, k=2 → 10^6   vs   k=3 → 10^9 (1000x plus lent!)\n\n")

# Récapitulatif des moyennes
cat("--- RÉCAPITULATIF DES MOYENNES DE k ---\n")
moyennes <- c(Cremer = mu_cremer, Quevy = mu_quevy, Lerat = mu_lerat)
for (nom in names(sort(moyennes))) {
  cat(sprintf("  %s : µ(k) = %.4f\n", nom, moyennes[nom]))
}
meilleur <- names(which.min(moyennes))
cat(sprintf("\n→ MEILLEUR ALGORITHME (k le plus faible) : %s\n", meilleur))


# -----------------------------------------------------------------------------
# 10. GRAPHIQUE FINAL DE SYNTHÈSE
# (boxplot pour comparer visuellement les 3 distributions)
# -----------------------------------------------------------------------------

p_final <- ggplot(algos_long, aes(x = algorithme, y = k, fill = algorithme)) +
  geom_boxplot(alpha = 0.7, outlier.color = "red") +
  geom_jitter(width = 0.1, alpha = 0.3, size = 1.5) +
  labs(
    title = "Comparaison des distributions de k (complexité algorithmique)",
    subtitle = "time = a × input^k   |   Plus k est faible, plus l'algo est efficace",
    x = "Algorithme",
    y = "Valeur de k",
    caption = "Boîte = Q1-Q3  |  Trait = médiane  |  Points rouges = outliers"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

print(p_final)
ggsave("synthese_boxplot.png", p_final, width = 8, height = 5)
cat("→ Fichier 'synthese_boxplot.png' sauvegardé\n\n")

cat("=== FIN DE L'EXERCICE ===\n")
cat("Fichiers générés : scatter_k.png, histogrammes_k.png,\n")
cat("                   normal_fit_k.png, synthese_boxplot.png\n")