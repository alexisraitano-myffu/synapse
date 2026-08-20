"""SYN-171 — harnais de parité modèles, à deux étages.

Étage 1 (`gate`)   : ~12 cas, quelques minutes, cherche les vices rédhibitoires
                     et s'arrête au premier. Ne mesure PAS la qualité.
Étage 2 (`full`)   : les 6 prompts du cycle, vérité-terrain étiquetée.

Le corpus committé ici est **100 % synthétique** : ce dépôt est public. Les
captures réelles restent accessibles en local via `--include-real`, jamais
versionnées.
"""
