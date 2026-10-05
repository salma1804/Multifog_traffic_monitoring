"""
model.py : modèle local de risque d'accident (tâche "inference" du nœud fog).

Ce que fait ce fichier
----------------------
1. Utilise TON preprocessing.py (mêmes features, même split temporel train / test).
2. Entraîne un classifieur sur la période d'entraînement (avant TRAIN_END).
3. Choisit un SEUIL d'alerte sur une partie récente du train (jamais sur le test).
4. Évalue sur la période de test, avec des métriques adaptées aux événements rares
   (les accidents sont ~2 % des lignes : l'accuracy ne veut rien dire).
5. Sauvegarde le modèle, la liste exacte des colonnes et le seuil dans un seul fichier.

Utilisation
-----------
    python model.py                 # entraîne (RandomForest par défaut) et sauvegarde
    python model.py --compare       # compare 3 modèles
    python model.py --ablation      # montre quelles informations le modèle utilise vraiment

Dans le code du nœud :
    from model import RiskModel
    rm = RiskModel()                    # charge le fichier sauvegardé
    result = rm.predict(X_nouvelles_lignes)   # risk, alert, level
"""
from __future__ import annotations

import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import NODE_CONFIG
from preprocessing import build_features, load_data, temporal_split

warnings.filterwarnings("ignore")
MODEL_PATH = f"model_{NODE_CONFIG['node_id']}.joblib"

# Modèle par défaut : RandomForest. Sur le test il a la meilleure précision pour un même niveau de
# détection (3 modèles comparés avec --compare), il n'a pas besoin de lot de données pour décider,
# et il est simple à expliquer. Les 3 modèles sont à égalité sur la mesure globale (PR-AUC).
DEFAULT_MODEL = "random_forest"


# --------------------------------------------------------------------------- #
# Les modèles possibles
# --------------------------------------------------------------------------- #
def make_model(name: str = "hist_gb"):
    """class_weight='balanced' : une erreur sur un accident (rare) coûte plus cher qu'une fausse alerte."""
    if name == "hist_gb":
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                              class_weight="balanced", random_state=0)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=0)
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    raise ValueError(f"modèle inconnu : {name}")


# --------------------------------------------------------------------------- #
# Seuil et métriques
# --------------------------------------------------------------------------- #
def choose_threshold(y_true, proba, beta: float = 2.0) -> float:
    """Seuil qui maximise le F-bêta. beta=2 : rater un accident est pire qu'une fausse alerte."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta**2) * prec * rec / np.maximum(beta**2 * prec + rec, 1e-12)
    return float(thr[int(np.argmax(f))])


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = proba >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "pr_auc_hasard": float(np.mean(y_true)),     # ce qu'un modèle qui devine obtiendrait
        "seuil": threshold,
        "recall": tp / max(tp + fn, 1),              # part des accidents détectés
        "precision": tp / max(tp + fp, 1),           # part des alertes qui sont de vrais accidents
        "alertes_%": 100 * float(np.mean(pred)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def risk_level(p: float, threshold: float) -> str:
    if p >= threshold:
        return "ALERTE"
    return "SURVEILLANCE" if p >= threshold / 2 else "NORMAL"


# --------------------------------------------------------------------------- #
# Entraînement
# --------------------------------------------------------------------------- #
def train(model_name: str = DEFAULT_MODEL, save: bool = True) -> dict:
    X, y, meta = build_features(load_data())
    Xtr, Xte, ytr, yte, mtr, _ = temporal_split(X, y, meta)

    # Seuil choisi sur les 20 % les plus RÉCENTS du train (validation temporelle, pas le test)
    cut = mtr["timestamp"].quantile(0.8)
    fit_mask = (mtr["timestamp"] <= cut).values
    probe = make_model(model_name).fit(Xtr[fit_mask], ytr[fit_mask])
    threshold = choose_threshold(ytr[~fit_mask], probe.predict_proba(Xtr[~fit_mask])[:, 1])

    model = make_model(model_name).fit(Xtr, ytr)          # modèle final : tout le train
    metrics = evaluate(yte, model.predict_proba(Xte)[:, 1], threshold)

    if save:
        joblib.dump({"model": model, "columns": list(X.columns), "threshold": threshold,
                     "node_id": NODE_CONFIG["node_id"], "model_name": model_name,
                     "metrics": metrics}, MODEL_PATH)
    return metrics


# --------------------------------------------------------------------------- #
# Utilisation par le nœud
# --------------------------------------------------------------------------- #
class RiskModel:
    """Charge le modèle sauvegardé. Les colonnes sont ALIGNÉES sur celles de l'entraînement :
    une catégorie absente (ex. 'Fog' jamais vu pendant une heure) devient une colonne à 0,
    donc le nœud ne plante pas et les colonnes restent identiques d'un nœud à l'autre."""

    def __init__(self, path: str = MODEL_PATH):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.columns = bundle["columns"]
        self.threshold = bundle["threshold"]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X.reindex(columns=self.columns, fill_value=0))[:, 1]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_proba(X)
        return pd.DataFrame({
            "risk": p,
            "alert": p >= self.threshold,
            "level": [risk_level(v, self.threshold) for v in p],
        }, index=X.index)


# --------------------------------------------------------------------------- #
# Comparaisons pédagogiques
# --------------------------------------------------------------------------- #
def compare() -> pd.DataFrame:
    rows = {name: train(name, save=False) for name in ("hist_gb", "random_forest", "logreg")}
    cols = ["roc_auc", "pr_auc", "recall", "precision", "alertes_%"]
    return pd.DataFrame(rows).T[cols].round(3)


def ablation() -> pd.DataFrame:
    """Quelles informations le modèle utilise-t-il vraiment ? On lui retire des familles de colonnes."""
    X, y, meta = build_features(load_data())
    Xtr, Xte, ytr, yte, _, _ = temporal_split(X, y, meta)
    now = ["vehicle_count", "average_speed", "heavy_vehicle_count", "motorcycle_count",
           "public_transport_count", "traffic_density", "queue_length", "average_wait_time",
           "parking_occupancy"]
    evolution = [c for c in X.columns if c.endswith(("_diff1", "_ratio"))]
    past = [c for c in X.columns if c.endswith("_roll3")]
    context = [c for c in X.columns if c not in now + evolution + past]
    groups = {
        "tout": list(X.columns),
        "contexte seul (météo, heure, route...)": context,
        "contexte + 3 heures PRÉCÉDENTES": context + past,
        "contexte + mesures de l'heure": context + now,
    }
    rows = {}
    for label, cols in groups.items():
        m = make_model("hist_gb").fit(Xtr[cols], ytr)
        p = m.predict_proba(Xte[cols])[:, 1]
        rows[label] = {"colonnes": len(cols), "roc_auc": roc_auc_score(yte, p),
                       "pr_auc": average_precision_score(yte, p)}
    out = pd.DataFrame(rows).T.round(3)
    out.loc["(hasard)"] = [np.nan, 0.5, round(float(yte.mean()), 3)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Modèle local de risque d'accident")
    ap.add_argument("--compare", action="store_true", help="compare 3 modèles")
    ap.add_argument("--ablation", action="store_true", help="quelles colonnes comptent vraiment")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=["hist_gb", "random_forest", "logreg"])
    args = ap.parse_args()

    if args.compare:
        print(compare().to_string())
    elif args.ablation:
        print(ablation().to_string())
    else:
        m = train(args.model)
        print(f"Modèle {args.model} sauvegardé -> {MODEL_PATH}\n")
        print(f"ROC-AUC  : {m['roc_auc']:.3f}")
        print(f"PR-AUC   : {m['pr_auc']:.3f}   (un modèle qui devine : {m['pr_auc_hasard']:.3f})")
        print(f"Seuil d'alerte : {m['seuil']:.3f}")
        print(f"Accidents détectés (recall) : {100*m['recall']:.0f} %  ({m['tp']} sur {m['tp'] + m['fn']})")
        print(f"Alertes justes (precision)  : {100*m['precision']:.0f} %  ({m['tp']} sur {m['tp'] + m['fp']})")
        print(f"Part des heures en alerte   : {m['alertes_%']:.1f} %")