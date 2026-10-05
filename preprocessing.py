# preprocessing.py charge ton CSV, garde les intersections de ta zone, supprime les 
# colonnes inutiles ou à fuite de données, et encode le texte en chiffres. Il crée aussi 
# des features d'évolution (variation, moyenne des 3 dernières heures, ratio) pour détecter 
# les changements brutaux, puis sépare les données en train (avant le 1er mars) et test (après).




"""Étape 2 : chargement, nettoyage et création des features (identique sur tous les noeuds)."""
import numpy as np
import pandas as pd
from config import NODE_CONFIG, TARGET, DROP_COLS, CATEGORICAL, DYNAMIC, TRAIN_END


def load_data(cfg=NODE_CONFIG) -> pd.DataFrame:
    df = pd.read_csv(cfg["csv_path"], parse_dates=["timestamp"])
    df = df[df["intersection_id"].isin(cfg["intersections"])]
    return df.sort_values(["intersection_id", "timestamp"]).reset_index(drop=True)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features dynamiques calculées PAR intersection, uniquement avec le passé."""
    g = df.groupby("intersection_id", group_keys=False)
    for col in DYNAMIC:
        df[f"{col}_diff1"] = g[col].diff().fillna(0)                      # variation vs heure précédente
        roll = g[col].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        df[f"{col}_roll3"] = roll.fillna(df[col])                         # moyenne des 3 h précédentes
        df[f"{col}_ratio"] = df[col] / (df[f"{col}_roll3"] + 1e-6)        # >1 = hausse brutale
    # heure cyclique (23h est proche de 0h)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def build_features(df: pd.DataFrame):
    """Retourne X (features), y (cible), meta (intersection_id + timestamp pour la suite)."""
    df = add_temporal_features(df.copy())
    meta = df[["intersection_id", "timestamp"]].copy()
    y = df[TARGET]
    X = df.drop(columns=DROP_COLS + [TARGET, "intersection_id", "timestamp", "latitude", "longitude", "road_id"])
    X = pd.get_dummies(X, columns=CATEGORICAL, dtype=int)
    return X, y, meta


def temporal_split(X, y, meta):
    train = (meta["timestamp"] < TRAIN_END).values
    return X[train], X[~train], y[train], y[~train], meta[train], meta[~train]


if __name__ == "__main__":
    df = load_data()
    X, y, meta = build_features(df)
    Xtr, Xte, ytr, yte, _, _ = temporal_split(X, y, meta)
    print(f"Noeud {NODE_CONFIG['node_id']} | zone {NODE_CONFIG['zone_name']}")
    print("Intersections :", sorted(df['intersection_id'].unique()))
    print(f"Features : {X.shape[1]} colonnes, {X.shape[0]} lignes")
    print(f"Train : {len(Xtr)} lignes, {ytr.sum()} accidents ({ytr.mean()*100:.2f}%)")
    print(f"Test  : {len(Xte)} lignes, {yte.sum()} accidents ({yte.mean()*100:.2f}%)")
    print("NaN restants :", int(X.isna().sum().sum()))
    print(list(X.columns))