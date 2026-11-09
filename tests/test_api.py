import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))

import pytest
from fastapi.testclient import TestClient

from app import ARTIFACTS, app

client = TestClient(app)
DUEL = {"pokemon_premier": 1, "pokemon_second": 4}


@pytest.fixture(scope="session", autouse=True)
def trained_models():
    """Entraîne les modèles Pokémon une fois pour toute la session de tests."""
    metrics_path = ARTIFACTS / "metrics.json"
    model_path = ARTIFACTS / "models.joblib"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else {}
    )
    if not model_path.exists() or metrics.get("schema_version") != "pokemon-duel-v1":
        from train import train
        train()


def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Pokémon Duel MLOps" in response.text
    assert "fruits" not in response.text.lower()


def test_health_and_pokemon_list():
    assert client.get("/api/health").json()["models_ready"] is True
    pokemon = client.get("/api/pokemon")
    assert pokemon.status_code == 200
    assert len(pokemon.json()) == 800
    assert pokemon.json()[0]["NUMERO"] == 1
    assert pokemon.json()[0]["NOM"] == "Bulbasaur"


def test_metrics_compare_three_classifiers():
    metrics = client.get("/api/metrics").json()
    assert metrics["n_combats"] == 50_000
    assert metrics["n_pokemon"] == 800
    assert set(metrics["classification"]["resultats"]) == {
        "regression_logistique", "random_forest", "reseau_neurones_mlp",
    }
    for result in metrics["classification"]["resultats"].values():
        assert 0.5 <= result["accuracy"] <= 1
        assert 0.5 <= result["f1_macro"] <= 1


def test_predict_returns_two_probabilities_and_recommendation():
    response = client.post("/api/predict", json=DUEL)
    assert response.status_code == 200
    result = response.json()
    assert result["pokemon_premier"]["numero"] == DUEL["pokemon_premier"]
    assert result["pokemon_second"]["numero"] == DUEL["pokemon_second"]
    assert result["recommandation"]["numero"] in DUEL.values()
    assert 0.5 <= result["confiance"] <= 1
    assert abs(result["probabilite_premier"] + result["probabilite_second"] - 1) <= 0.002
    reversed_result = client.post("/api/predict", json={
        "pokemon_premier": DUEL["pokemon_second"],
        "pokemon_second": DUEL["pokemon_premier"],
    }).json()
    assert abs(
        result["probabilite_premier"] - reversed_result["probabilite_second"]
    ) <= 0.002


def test_predict_validates_ids():
    assert client.post("/api/predict", json={
        "pokemon_premier": 1,
        "pokemon_second": 1,
    }).status_code == 400
    assert client.post("/api/predict", json={
        "pokemon_premier": 1,
        "pokemon_second": 9999,
    }).status_code == 404


def test_batch_prediction_returns_enriched_csv():
    csv = "POKEMON_PREMIER,POKEMON_SECOND\n1,4\n7,9\n"
    response = client.post(
        "/api/predict-batch",
        files={"file": ("duels.csv", csv, "text/csv")},
    )
    assert response.status_code == 200
    assert "pokemon_recommande" in response.text.splitlines()[0]
    assert "numero_recommande" in response.text.splitlines()[0]
    assert len(response.text.splitlines()) == 3


def test_batch_rejects_bad_schema():
    response = client.post(
        "/api/predict-batch",
        files={"file": ("bad.csv", "a,b\n1,2\n", "text/csv")},
    )
    assert response.status_code == 400
    assert "Colonnes manquantes" in response.json()["detail"]


def test_charts_are_available():
    for name in ("confusion_matrix", "feature_importance", "learning_curve"):
        response = client.get(f"/api/charts/{name}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
    assert client.get("/api/charts/inconnu").status_code == 404


def test_history_and_drift_monitoring():
    for _ in range(6):
        response = client.post("/api/predict", json=DUEL)
        assert response.status_code == 200
    history = client.get("/api/history").json()
    assert len(history) >= 6
    assert history[0]["pokemon_premier"]["numero"] == DUEL["pokemon_premier"]

    monitoring = client.get("/api/monitoring").json()
    assert monitoring["statut"] in ("ok", "alerte dérive")
    assert len(monitoring["drift"]) == 9


def test_training_metrics_artifact_is_pokemon_pipeline():
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == "pokemon-duel-v1"
    assert metrics["features"]
