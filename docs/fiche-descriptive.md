# Fiche descriptive — Pokémon Duel MLOps

## Présentation

Application web de démonstration qui estime lequel de deux Pokémon a le plus de chances de gagner, puis présente la recommandation et son niveau de confiance. Elle inclut une API, des prédictions individuelles ou en lot, une comparaison de modèles et un écran de suivi.

## Besoin et objectif

Le projet reçu affichait un exemple de fruits alors que les données et le notebook décrivent des Pokémon. L'objectif corrigé est de recommander au dresseur un Pokémon pour un duel, à partir de résultats historiques.

## Données d'entrée

| Fichier | Contenu utilisé |
|---|---|
| `Donné/pokedex.csv` | 800 Pokémon, identifiants, types et statistiques |
| `Donné/combats.csv` | 50 000 duels et identifiant du gagnant |
| `Donné/poke_type_chart.csv` | Multiplicateurs d'efficacité entre types |

`Donné/dataset.csv` est un fichier agrégé issu de l'exploration du notebook; le pipeline de duel repart des données du Pokédex et des combats pour éviter d'apprendre le taux global d'un Pokémon comme s'il s'agissait du résultat d'un duel précis.

## Pipeline de traitement IA

1. Chargement et validation des fichiers CSV.
2. Construction des différences de points de vie, attaque, défense, attaque spéciale, défense spéciale, vitesse et statut légendaire; calcul d'un indicateur d'efficacité des types.
3. Séparation stratifiée entre apprentissage et test. Les duels d'apprentissage sont aussi présentés dans l'ordre inverse pour ne pas privilégier la position « premier Pokémon ».
4. Entraînement comparatif de la régression logistique, du Random Forest et d'un réseau de neurones MLP.
5. Sélection sur le F1 macro du test, export des métriques et génération des graphiques.
6. Déploiement de l'application FastAPI dans un conteneur Docker. Les endpoints API et l'interface web sont servis par le même service.

## Résultats de l'entraînement livré

Sur l'entraînement reproductible fourni (`random_state=42`), 37 500 combats servent à l'apprentissage et 12 500 au test. L'évaluation équilibre l'ordre des deux participants.

| Modèle | Accuracy | F1 macro |
|---|---:|---:|
| Régression logistique | 88,47 % | 88,47 % |
| Random Forest (retenu) | 93,77 % | 93,77 % |
| Réseau de neurones MLP | 93,47 % | 93,47 % |

## Utilisation

- En local : <http://127.0.0.1:8000>
- Documentation API : <http://127.0.0.1:8000/docs>
- Santé du service : <http://127.0.0.1:8000/api/health>
- URL publique : à renseigner après création du service Render depuis le dépôt du projet.

## Sorties

- Pokémon recommandé et probabilités estimées pour chaque participant.
- CSV enrichi pour les prédictions en lot.
- Accuracy, F1 macro, matrice de confusion, importance des caractéristiques et courbe d'apprentissage.
- Historique récent et alerte indicative de dérive des données.

## Limites

La recommandation reflète les données fournies et les caractéristiques présentes. Le système ne simule pas les attaques, niveaux, objets, capacités, conditions de combat ni stratégies. La confiance affichée est une probabilité estimée par le modèle, pas une garantie de victoire. Les métriques réelles doivent être consultées sur `/api/metrics` après l'entraînement.

## Lancement reproductible

Voir le [README du projet](../README.md) pour l'installation Windows, l'entraînement, les tests et le déploiement Docker/Render.
