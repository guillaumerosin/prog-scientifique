# setup.jl A exécuter UNE SEULE FOIS pour installer les dépendances
using Pkg
Pkg.activate("..")          # remonte à la racine du projet
Pkg.add(["CSV", "DataFrames", "Statistics", "Plots"])