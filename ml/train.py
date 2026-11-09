"""Entraîne et compare des classifieurs pour recommander un Pokémon en duel."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import (
    ARTIFACTS_DIR,
    FEATURES,
    build_battle_dataset,
    load_data,
    reverse_pair_features,
    symmetric_first_probability,
)

CHARTS = ARTIFACTS_DIR / "charts"
ARTIFACT_VERSION = "pokemon-duel-v1"

CLASSIFIERS = {
    "regression_logistique": LogisticRegression(max_iter=1500, random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=160, min_samples_leaf=2, n_jobs=-1, random_state=42,
    ),
    "reseau_neurones_mlp": MLPClassifier(
        hidden_layer_sizes=(48, 24),
        activation="relu",
        max_iter=250,
        early_stopping=True,
        random_state=42,
    ),
}


def _augment_pairs(
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, np.ndarray]:
    reversed_features = reverse_pair_features(features)
    return (
        pd.concat([features, reversed_features], ignore_index=True),
        np.concatenate([labels.to_numpy(), 1 - labels.to_numpy()]),
    )


def train() -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    pokedex, battles, type_chart = load_data()
    all_features, all_labels = build_battle_dataset(battles, pokedex, type_chart)
    train_indices, test_indices = train_test_split(
        np.arange(len(battles)),
        test_size=0.25,
        random_state=42,
        stratify=all_labels,
    )

    raw_train_features = all_features.iloc[train_indices].reset_index(drop=True)
    raw_train_labels = all_labels.iloc[train_indices].reset_index(drop=True)
    X_train, y_train = _augment_pairs(raw_train_features, raw_train_labels)

    raw_test_features = all_features.iloc[test_indices].reset_index(drop=True)
    raw_test_labels = all_labels.iloc[test_indices].reset_index(drop=True)
    X_test, y_test = _augment_pairs(raw_test_features, raw_test_labels)

    results = {}
    best_name = None
    best_model = None
    best_f1 = -1.0
    fitted_models = {}
    for name, classifier in CLASSIFIERS.items():
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])
        model.fit(X_train, y_train)
        predictions = (np.array(symmetric_first_probability(model, X_test)) >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="macro")
        results[name] = {
            "accuracy": round(float(accuracy), 4),
            "f1_macro": round(float(f1), 4),
        }
        fitted_models[name] = model
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    predictions = (
        np.array(symmetric_first_probability(best_model, X_test)) >= 0.5
    ).astype(int)
    confusion = confusion_matrix(y_test, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(
        confusion,
        display_labels=["Deuxième gagne", "Premier gagne"],
    ).plot(ax=axis, colorbar=False, cmap="Blues", xticks_rotation=15)
    axis.set_title(f"Matrice de confusion — {best_name}")
    figure.tight_layout()
    figure.savefig(CHARTS / "confusion_matrix.png", dpi=110)
    plt.close(figure)

    random_forest = fitted_models["random_forest"].named_steps["classifier"]
    importances = random_forest.feature_importances_
    order = np.argsort(importances)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.barh(np.array(FEATURES)[order], importances[order], color="#1d4ed8")
    axis.set_title("Importance des caractéristiques (Random Forest)")
    figure.tight_layout()
    figure.savefig(CHARTS / "feature_importance.png", dpi=110)
    plt.close(figure)

    curve_size = min(6000, len(X_train))
    curve_features = X_train.iloc[:curve_size]
    curve_labels = y_train[:curve_size]
    sizes, train_scores, validation_scores = learning_curve(
        best_model,
        curve_features,
        curve_labels,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        train_sizes=[0.25, 0.6, 1.0],
        n_jobs=-1,
    )
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(sizes, train_scores.mean(axis=1), "o-", label="entraînement")
    axis.plot(sizes, validation_scores.mean(axis=1), "s-", label="validation")
    axis.set_xlabel("Taille du jeu d'entraînement")
    axis.set_ylabel("Accuracy")
    axis.set_title(f"Courbe d'apprentissage — {best_name}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(CHARTS / "learning_curve.png", dpi=110)
    plt.close(figure)

    metrics = {
        "schema_version": ARTIFACT_VERSION,
        "application": "Pokémon Duel MLOps",
        "objectif": "Recommander le gagnant d'un duel entre deux Pokémon.",
        "classification": {
            "modele_retenu": best_name,
            "resultats": results,
        },
        "n_pokemon": len(pokedex),
        "n_combats": len(battles),
        "n_combats_entrainement": len(train_indices),
        "n_combats_test": len(test_indices),
        "features": FEATURES,
        "moyennes_entrainement": X_train.mean().round(4).to_dict(),
        "ecarts_types_entrainement": X_train.std().replace(0, 1).round(4).to_dict(),
    }
    joblib.dump(
        {"schema_version": ARTIFACT_VERSION, "classifier": best_model},
        ARTIFACTS_DIR / "models.joblib",
    )
    (ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2, ensure_ascii=False))
