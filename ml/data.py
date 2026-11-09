"""Chargement et préparation des données de combats Pokémon."""
from math import prod
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Donné"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

STAT_FEATURES = [
    "POINTS_DE_VIE",
    "NIVEAU_ATTAQUE",
    "NIVEAU_DEFENSE",
    "NIVEAU_ATTAQUE_SPECIALE",
    "NIVEAU_DEFENSE_SPECIALE",
    "VITESSE",
]
FEATURES = [
    *(f"diff_{feature.lower()}" for feature in STAT_FEATURES),
    "diff_legendaire",
    "efficacite_type_premier",
    "efficacite_type_second",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge le Pokédex, les combats historiques et le tableau des types."""
    pokedex = pd.read_csv(DATA_DIR / "pokedex.csv")
    pokedex.columns = [column.strip() for column in pokedex.columns]

    required_pokemon_columns = {
        "NUMERO", "NOM", "TYPE_1", "TYPE_2", "LEGENDAIRE", *STAT_FEATURES,
    }
    missing = required_pokemon_columns - set(pokedex.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans pokedex.csv : {', '.join(sorted(missing))}")

    pokedex["NOM"] = pokedex["NOM"].fillna(
        pokedex["NUMERO"].map(lambda number: f"Pokémon #{int(number)}")
    )
    for column in ("TYPE_1", "TYPE_2"):
        pokedex[column] = pokedex[column].fillna("").astype(str).str.strip()
    pokedex["LEGENDAIRE"] = (
        pokedex["LEGENDAIRE"].astype(str).str.strip().str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
    )
    for column in ["NUMERO", *STAT_FEATURES, "LEGENDAIRE"]:
        pokedex[column] = pd.to_numeric(pokedex[column], errors="coerce")
    if pokedex[["NUMERO", *STAT_FEATURES, "LEGENDAIRE"]].isna().any().any():
        raise ValueError("Le Pokédex contient des statistiques ou identifiants invalides.")
    pokedex["NUMERO"] = pokedex["NUMERO"].astype(int)
    pokedex["LEGENDAIRE"] = pokedex["LEGENDAIRE"].astype(int)
    if pokedex["NUMERO"].duplicated().any():
        raise ValueError("Le Pokédex contient des identifiants Pokémon en double.")

    battles = pd.read_csv(DATA_DIR / "combats.csv")
    required_battle_columns = {"POKEMON_PREMIER", "POKEMON_SECOND", "GAGNANT"}
    missing = required_battle_columns - set(battles.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans combats.csv : {', '.join(sorted(missing))}")
    for column in required_battle_columns:
        battles[column] = pd.to_numeric(battles[column], errors="coerce")
    if battles[list(required_battle_columns)].isna().any().any():
        raise ValueError("combats.csv contient des identifiants de combat invalides.")
    battles[list(required_battle_columns)] = battles[list(required_battle_columns)].astype(int)

    pokedex_ids = set(pokedex["NUMERO"])
    participants = set(battles["POKEMON_PREMIER"]) | set(battles["POKEMON_SECOND"])
    winners = set(battles["GAGNANT"])
    if not participants <= pokedex_ids or not winners <= participants:
        raise ValueError("Les combats font référence à un Pokémon absent du Pokédex.")
    if not (
        battles["GAGNANT"].eq(battles["POKEMON_PREMIER"])
        | battles["GAGNANT"].eq(battles["POKEMON_SECOND"])
    ).all():
        raise ValueError("Chaque gagnant doit être l'un des deux Pokémon du combat.")

    type_chart = pd.read_csv(DATA_DIR / "poke_type_chart.csv")
    type_chart.columns = [column.strip() for column in type_chart.columns]
    type_chart = type_chart.set_index(type_chart.columns[0])
    type_chart.index = type_chart.index.astype(str).str.strip()
    type_chart.columns = type_chart.columns.astype(str).str.strip()

    return pokedex, battles, type_chart.astype(float)


def _type_effectiveness(
    attacker: dict,
    defender: dict,
    type_chart: pd.DataFrame,
) -> float:
    attacking_types = [attacker["TYPE_1"], attacker["TYPE_2"]]
    defending_types = [defender["TYPE_1"], defender["TYPE_2"]]
    attacking_types = [pokemon_type for pokemon_type in attacking_types if pokemon_type]
    defending_types = [pokemon_type for pokemon_type in defending_types if pokemon_type]
    if not attacking_types or not defending_types:
        return 1.0
    return max(
        prod(float(type_chart.loc[attack_type, defending_type]) for defending_type in defending_types)
        for attack_type in attacking_types
    )


def build_pair_features(
    first_id: int,
    second_id: int,
    pokedex: pd.DataFrame,
    type_chart: pd.DataFrame,
    pokemon_by_id: dict[int, dict] | None = None,
) -> dict[str, float]:
    """Construit les caractéristiques comparatives d'un duel."""
    if pokemon_by_id is None:
        pokemon_by_id = pokedex.set_index("NUMERO").to_dict(orient="index")
    return _build_pair_features(first_id, second_id, pokemon_by_id, type_chart)


def _build_pair_features(
    first_id: int,
    second_id: int,
    pokemon_by_id: dict[int, dict],
    type_chart: pd.DataFrame,
) -> dict[str, float]:
    try:
        first = pokemon_by_id[int(first_id)]
        second = pokemon_by_id[int(second_id)]
    except KeyError as error:
        raise ValueError(f"Pokémon inconnu : {error.args[0]}") from error

    features = {
        f"diff_{feature.lower()}": float(first[feature] - second[feature])
        for feature in STAT_FEATURES
    }
    features["diff_legendaire"] = float(first["LEGENDAIRE"] - second["LEGENDAIRE"])
    features["efficacite_type_premier"] = _type_effectiveness(first, second, type_chart)
    features["efficacite_type_second"] = _type_effectiveness(second, first, type_chart)
    return features


def build_battle_dataset(
    battles: pd.DataFrame,
    pokedex: pd.DataFrame,
    type_chart: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Transforme les combats étiquetés en caractéristiques et cible binaire."""
    pokemon_by_id = pokedex.set_index("NUMERO").to_dict(orient="index")
    features = [
        _build_pair_features(first, second, pokemon_by_id, type_chart)
        for first, second in battles[
            ["POKEMON_PREMIER", "POKEMON_SECOND"]
        ].itertuples(index=False, name=None)
    ]
    labels = battles["GAGNANT"].eq(battles["POKEMON_PREMIER"]).astype(int)
    return pd.DataFrame(features, columns=FEATURES), labels.reset_index(drop=True)


def reverse_pair_features(features: pd.DataFrame) -> pd.DataFrame:
    """Retourne l'ordre du duel : inverse les différences et échange les types."""
    reversed_features = features.copy()
    for column in FEATURES:
        if column.startswith("diff_"):
            reversed_features[column] = -features[column]
    reversed_features["efficacite_type_premier"] = features["efficacite_type_second"]
    reversed_features["efficacite_type_second"] = features["efficacite_type_premier"]
    return reversed_features


def symmetric_first_probability(
    classifier: Pipeline,
    features: pd.DataFrame,
) -> list[float]:
    """Moyenne les deux ordres du duel pour neutraliser le biais de position."""
    positive_class = list(classifier.classes_).index(1)
    direct_probability = classifier.predict_proba(features)[:, positive_class]
    reverse_probability = classifier.predict_proba(
        reverse_pair_features(features)
    )[:, positive_class]
    return ((direct_probability + 1 - reverse_probability) / 2).tolist()


def list_pokemon(pokedex: pd.DataFrame) -> list[dict[str, int | str]]:
    """Retourne les champs d'identification nécessaires aux menus de l'interface."""
    return (
        pokedex[["NUMERO", "NOM", "TYPE_1", "TYPE_2"]]
        .sort_values("NUMERO")
        .to_dict(orient="records")
    )


if __name__ == "__main__":
    pokemon, combats, _ = load_data()
    print(f"{len(pokemon)} Pokémon et {len(combats)} combats chargés depuis {DATA_DIR}")
