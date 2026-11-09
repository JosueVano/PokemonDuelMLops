"""API et interface web pour recommander un Pokémon lors d'un duel."""
import io
import json
import sys
import threading
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ml"))

from data import (
    ARTIFACTS_DIR,
    FEATURES,
    build_pair_features,
    list_pokemon,
    load_data,
    symmetric_first_probability,
)

ARTIFACTS = ARTIFACTS_DIR
HISTORY_PATH = ARTIFACTS / "pokemon-history.json"

app = FastAPI(
    title="Pokémon Duel MLOps",
    description="Comparaison de modèles IA pour recommander un Pokémon en duel.",
)
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

_bundle = None
_lock = threading.Lock()
_pokemon_data = None
_pokemon_index = None
ARTIFACT_VERSION = "pokemon-duel-v1"


def get_pokemon_data():
    global _pokemon_data, _pokemon_index
    if _pokemon_data is None:
        _pokemon_data = load_data()
        _pokemon_index = _pokemon_data[0].set_index("NUMERO").to_dict(orient="index")
    return _pokemon_data


def get_models():
    global _bundle
    if _bundle is None:
        model_path = ARTIFACTS / "models.joblib"
        metrics_path = ARTIFACTS / "metrics.json"
        if model_path.exists():
            saved_bundle = joblib.load(model_path)
        else:
            saved_bundle = {}
        saved_metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.exists()
            else {}
        )
        if (
            saved_bundle.get("schema_version") != ARTIFACT_VERSION
            or saved_metrics.get("schema_version") != ARTIFACT_VERSION
        ):
            from train import train
            train()
        _bundle = joblib.load(model_path)
    return _bundle


def get_metrics() -> dict:
    get_models()
    path = ARTIFACTS / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_history() -> list:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    return []


def save_history(entries: list) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(
        json.dumps(entries[-500:], ensure_ascii=False),
        encoding="utf-8",
    )


class DuelIn(BaseModel):
    pokemon_premier: int = Field(gt=0)
    pokemon_second: int = Field(gt=0)


def predict_duel(duel: DuelIn) -> dict:
    pokedex, _, type_chart = get_pokemon_data()
    pokemon_by_id = pokedex.set_index("NUMERO")
    first_id, second_id = duel.pokemon_premier, duel.pokemon_second
    if first_id == second_id:
        raise HTTPException(400, "Choisissez deux Pokémon différents.")
    if first_id not in pokemon_by_id.index or second_id not in pokemon_by_id.index:
        raise HTTPException(404, "Identifiant Pokémon introuvable dans le Pokédex.")

    classifier = get_models()["classifier"]
    metrics = get_metrics()
    features = build_pair_features(
        first_id,
        second_id,
        pokedex,
        type_chart,
        _pokemon_index,
    )
    X = pd.DataFrame([features], columns=metrics["features"])
    first_probability = float(symmetric_first_probability(classifier, X)[0])
    second_probability = 1.0 - first_probability
    winner_id = first_id if first_probability >= 0.5 else second_id
    winner_probability = max(first_probability, second_probability)

    first_name = str(pokemon_by_id.loc[first_id, "NOM"])
    second_name = str(pokemon_by_id.loc[second_id, "NOM"])
    winner_name = first_name if winner_id == first_id else second_name
    return {
        "pokemon_premier": {"numero": first_id, "nom": first_name},
        "pokemon_second": {"numero": second_id, "nom": second_name},
        "probabilite_premier": round(first_probability, 3),
        "probabilite_second": round(second_probability, 3),
        "recommandation": {"numero": winner_id, "nom": winner_name},
        "confiance": round(winner_probability, 3),
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/health")
def health():
    metrics_path = ARTIFACTS / "metrics.json"
    models_ready = (
        (ARTIFACTS / "models.joblib").exists()
        and metrics_path.exists()
        and json.loads(metrics_path.read_text(encoding="utf-8")).get("schema_version")
        == ARTIFACT_VERSION
    )
    return {
        "status": "ok",
        "app": "Pokémon Duel MLOps",
        "models_ready": models_ready,
    }


@app.get("/api/pokemon")
def pokemon():
    pokedex = get_pokemon_data()[0]
    return list_pokemon(pokedex)


@app.get("/api/metrics")
def metrics():
    return get_metrics()


@app.post("/api/predict")
def predict(duel: DuelIn):
    result = predict_duel(duel)
    history = load_history()
    history.append({
        "horodatage": datetime.now().isoformat(timespec="seconds"),
        "pokemon_premier": result["pokemon_premier"],
        "pokemon_second": result["pokemon_second"],
        "recommandation": result["recommandation"],
        "confiance": result["confiance"],
    })
    save_history(history)
    return result


@app.post("/api/predict-batch")
async def predict_batch(file: UploadFile):
    """Ajoute les recommandations d'un lot de duels au CSV reçu."""
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise HTTPException(400, "CSV illisible ou vide.") from error
    required = ["POKEMON_PREMIER", "POKEMON_SECOND"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise HTTPException(400, f"Colonnes manquantes : {', '.join(missing)}")
    if len(df) > 10_000:
        raise HTTPException(400, "Maximum 10 000 duels par fichier.")
    if df.empty:
        raise HTTPException(400, "Le fichier ne contient aucun duel.")

    for column in required:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.isna().any() or not numeric.mod(1).eq(0).all():
            raise HTTPException(400, f"Identifiants invalides dans la colonne {column}.")
        df[column] = numeric.astype(int)

    predictions = [
        predict_duel(DuelIn(
            pokemon_premier=row.POKEMON_PREMIER,
            pokemon_second=row.POKEMON_SECOND,
        ))
        for row in df[required].itertuples(index=False)
    ]
    df["pokemon_recommande"] = [item["recommandation"]["nom"] for item in predictions]
    df["numero_recommande"] = [item["recommandation"]["numero"] for item in predictions]
    df["confiance"] = [item["confiance"] for item in predictions]
    df["probabilite_premier"] = [item["probabilite_premier"] for item in predictions]

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="duels_recommandes.csv"'},
    )


@app.get("/api/charts/{name}")
def chart(name: str):
    allowed = {"confusion_matrix", "feature_importance", "learning_curve"}
    if name not in allowed:
        raise HTTPException(404)
    path = ARTIFACTS / "charts" / f"{name}.png"
    if not path.exists():
        raise HTTPException(404, "Graphique pas encore généré — entraîner d'abord.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/history")
def history():
    return list(reversed(load_history()))[:50]


@app.get("/api/monitoring")
def monitoring():
    """Compare les caractéristiques des duels récents au jeu d'entraînement."""
    metrics_data = get_metrics()
    entries = load_history()
    if len(entries) < 5:
        return {"statut": "pas assez de données", "n_predictions": len(entries), "drift": []}

    pokedex, _, type_chart = get_pokemon_data()
    recent_features = pd.DataFrame([
        build_pair_features(
            entry["pokemon_premier"]["numero"],
            entry["pokemon_second"]["numero"],
            pokedex,
            type_chart,
            _pokemon_index,
        )
        for entry in entries[-100:]
    ])
    drift = []
    for feature in FEATURES:
        mean_train = metrics_data["moyennes_entrainement"][feature]
        std_train = metrics_data["ecarts_types_entrainement"][feature] or 1
        z_score = abs(recent_features[feature].mean() - mean_train) / std_train
        drift.append({
            "feature": feature,
            "moyenne_entrainement": mean_train,
            "moyenne_recente": round(float(recent_features[feature].mean()), 2),
            "z_score": round(float(z_score), 2),
            "alerte": bool(z_score > 2),
        })
    return {
        "statut": "alerte dérive" if any(item["alerte"] for item in drift) else "ok",
        "n_predictions": len(entries),
        "drift": drift,
    }


@app.post("/api/retrain")
def retrain():
    """Réentraîne les classifieurs à partir des fichiers Pokémon présents."""
    global _bundle
    with _lock:
        from train import train
        result = train()
        _bundle = None
    return {"statut": "ré-entraîné", "classification": result["classification"]}
