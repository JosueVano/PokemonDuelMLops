# Pokémon Duel MLOps

Application web qui compare deux Pokémon et recommande celui qui a le plus de chances de gagner, à partir du Pokédex, du tableau des types et des combats historiques fournis dans `Donné/`.

Le notebook [`pipeline_pokemon.ipynb`](pipeline_pokemon.ipynb) présente et exécute le pipeline corrigé de bout en bout. Le notebook original [`Donné/note.ipynb`](Donné/note.ipynb) présente l'exploration initiale et l'objectif métier. L'application et le nouveau notebook utilisent les résultats individuels de `combats.csv` pour entraîner un classifieur de duel, plutôt que la régression sur le taux de victoire agrégé de l'ancien notebook.

## Démarrer en local (Windows)

Depuis PowerShell, placez-vous dans le dossier du projet :

```powershell
cd D:\Projet\ML-main
```

Avec Python 3.11 ou 3.12, créez un environnement virtuel et installez les dépendances :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell bloque l'activation du venv, exécutez la commande suivante dans la fenêtre courante, puis recommencez l'activation :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Dans VS Code, sélectionnez ensuite l'interpréteur du projet avec **Python: Select Interpreter** → `.venv\Scripts\python.exe`.

Entraînez les modèles, puis démarrez le serveur web :

```powershell
python ml\train.py
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

- Interface : <http://127.0.0.1:8000>
- Documentation interactive de l'API : <http://127.0.0.1:8000/docs>
- Vérification de santé : <http://127.0.0.1:8000/api/health>

Le premier entraînement crée `ml/artifacts/` (modèle, métriques et graphiques). Il peut prendre quelques minutes. L'API peut également entraîner automatiquement les modèles si les artefacts n'existent pas.

## Pipeline IA

1. **Données** — `Donné/pokedex.csv` (statistiques et types), `Donné/combats.csv` (vainqueur observé) et `Donné/poke_type_chart.csv` (efficacité des types).
2. **Préparation** — contrôle des identifiants et des valeurs, calcul des écarts de statistiques, du statut légendaire et des efficacités de types entre les deux combattants.
3. **Apprentissage** — séparation entraînement/test stratifiée, inversion des duels d'entraînement pour réduire le biais de position, puis comparaison entre régression logistique, Random Forest et MLP.
4. **Évaluation** — Accuracy et F1 macro sur le jeu de test, matrice de confusion, importance des caractéristiques et courbe d'apprentissage.
5. **Déploiement** — API FastAPI et interface web; historique, endpoint de prédiction en lot, détection de dérive et endpoint de ré-entraînement.

Le résultat est une recommandation statistique fondée sur ces données; ce n'est pas une simulation complète d'un combat Pokémon (attaques, niveaux, objets et stratégie ne sont pas modélisés).

## Déploiement Docker

Construisez et démarrez le conteneur :

```powershell
docker build -t pokemon-duel-mlops .
docker run --rm -p 8000:8000 pokemon-duel-mlops
```

Une configuration Render est disponible dans [`render.yaml`](render.yaml). Après avoir poussé le projet vers un dépôt GitHub accessible, créez un Blueprint Render à partir de ce dépôt. Render exécutera l'entraînement dans l'image Docker, surveillera `/api/health` et fournira l'URL publique du service après déploiement. Cette URL dépend du compte et du service Render; elle n'est pas créée depuis ce dépôt.

## API

- `GET /api/pokemon` — liste des Pokémon disponibles.
- `POST /api/predict` — prédit le vainqueur probable d'un duel (`pokemon_premier`, `pokemon_second`).
- `POST /api/predict-batch` — enrichit un CSV de duels (`POKEMON_PREMIER`, `POKEMON_SECOND`).
- `GET /api/metrics` — métriques et taille des jeux de données.
- `GET /api/monitoring` — indicateurs de dérive des caractéristiques récentes.
- `POST /api/retrain` — réentraîne et recharge les modèles.

## Vérification

```powershell
pytest -v
```

## Livrable

Le rapport complet à remettre est [`docs/rapport_pokemon_ml_dl.pdf`](docs/rapport_pokemon_ml_dl.pdf), avec la fiche synthétique dans [`docs/fiche-descriptive.md`](docs/fiche-descriptive.md). En local, l'URL de démonstration est `http://127.0.0.1:8000`; après publication, remplacez-la par l'URL publique fournie par Render.
