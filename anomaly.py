"""
anomaly.py : détection d'anomalies locale (tâche "anomaly" du nœud fog).

Différence avec model.py
------------------------
model.py répond à "y a-t-il un accident ?" (il a vu des accidents pendant l'entraînement).
anomaly.py répond à "est-ce que le trafic fait quelque chose d'inhabituel ?" sans avoir besoin
d'exemples d'accidents : il apprend ce qui est NORMAL et signale tout ce qui s'en écarte.
Il peut donc repérer une situation anormale même d'un genre jamais vu.

Deux détecteurs combinés :
    1. IsolationForest sur les variables d'ÉVOLUTION (variation, ratio) : donne un score d'étrangeté.
    2. Règles de seuils : une hausse ou une chute brutale d'une mesure (ratio hors des cas
       habituels du train). Elles servent surtout à EXPLIQUER l'alerte ("la file explose").

Utilisation
-----------
    python anomaly.py                    # entraîne, évalue sur le test, montre des exemples

Dans le code du nœud :
    from anomaly import AnomalyDetector
    det = AnomalyDetector.load()
    out = det.predict(X_nouvelles_lignes)    # score, is_anomaly, reasons
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from config import DYNAMIC, NODE_CONFIG
from preprocessing import build_features, load_data, temporal_split

warnings.filterwarnings("ignore")
ANOMALY_PATH = f"anomaly_{NODE_CONFIG['node_id']}.joblib"
GATE_QUANTILE = 0.70  # porte large : au-dessus de ce quantile du train, on appelle le modèle (voir node_engine.py)
RATIO_CLIP = 20.0     # un ratio peut exploser quand la moyenne passée est ~0 : on le plafonne

LABELS = {
    "queue_length": "la file d'attente",
    "average_wait_time": "le temps d'attente",
    "average_speed": "la vitesse moyenne",
    "vehicle_count": "le nombre de véhicules",
}


class AnomalyDetector:
    def __init__(self, score_quantile: float = 0.97, rule_quantile: float = 0.99, gate_quantile: float = GATE_QUANTILE):
        self.score_quantile = score_quantile    # 3 % des heures les plus étranges du train = anormales
        self.rule_quantile = rule_quantile
        self.gate_quantile = gate_quantile      # la "porte" : les 30 % les plus étranges passent au modèle

    # ---- préparation des colonnes d'évolution ----
    def _evolution(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [f"{c}_{s}" for c in DYNAMIC for s in ("diff1", "ratio")]
        E = X.reindex(columns=cols, fill_value=0).copy()
        for c in cols:
            if c.endswith("_ratio"):
                E[c] = E[c].clip(0, RATIO_CLIP)
        return E

    def fit(self, X_train: pd.DataFrame) -> "AnomalyDetector":
        E = self._evolution(X_train)
        self.columns = list(E.columns)
        self.iso = IsolationForest(n_estimators=200, random_state=0).fit(E)
        train_scores = -self.iso.score_samples(E)
        self.threshold = float(np.quantile(train_scores, self.score_quantile))
        self.gate_threshold = float(np.quantile(train_scores, self.gate_quantile))   # choisi sur le TRAIN, jamais sur le test
        # seuils de règle : une mesure est "brutale" si son ratio sort des cas habituels du train
        self.rules = {}
        for c in DYNAMIC:
            r = E[f"{c}_ratio"]
            self.rules[c] = (float(r.quantile(1 - self.rule_quantile)), float(r.quantile(self.rule_quantile)))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return -self.iso.score_samples(self._evolution(X))

    def _reasons(self, E: pd.DataFrame) -> list[str]:
        out = [[] for _ in range(len(E))]
        for c, (low, high) in self.rules.items():
            ratio = E[f"{c}_ratio"].to_numpy()
            for i, r in enumerate(ratio):
                if r > high:
                    out[i].append(f"{LABELS[c]} monte brutalement (x{r:.1f})")
                elif r < low:
                    out[i].append(f"{LABELS[c]} chute brutalement (x{r:.2f})")
        return ["; ".join(o) for o in out]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        E = self._evolution(X)
        s = -self.iso.score_samples(E)
        return pd.DataFrame({"score": s, "is_anomaly": s >= self.threshold, "gate_open": s >= self.gate_threshold,
                             "reasons": self._reasons(E)}, index=X.index)

    def save(self, path: str = ANOMALY_PATH) -> None:
        # On sauvegarde des DONNÉES (modèle sklearn, seuils...) et non l'objet Python lui-même :
        # un objet créé par `python anomaly.py` ne pourrait pas être rechargé par un autre programme
        # (api.py, pytest...) car Python l'enregistrerait comme appartenant au module "__main__".
        joblib.dump({"params": {"score_quantile": self.score_quantile, "rule_quantile": self.rule_quantile,
                                "gate_quantile": self.gate_quantile},
                     "columns": self.columns, "iso": self.iso, "threshold": self.threshold,
                     "gate_threshold": self.gate_threshold, "rules": self.rules}, path)

    @staticmethod
    def load(path: str = ANOMALY_PATH) -> "AnomalyDetector":
        b = joblib.load(path)
        if "gate_threshold" not in b:
            raise ValueError(f"{path} est un ancien fichier (sans la porte) : relance  python anomaly.py")
        det = AnomalyDetector(**b["params"])
        det.columns, det.iso, det.threshold, det.rules = b["columns"], b["iso"], b["threshold"], b["rules"]
        det.gate_threshold = b["gate_threshold"]
        return det


if __name__ == "__main__":
    X, y, meta = build_features(load_data())
    Xtr, Xte, ytr, yte, _, mte = temporal_split(X, y, meta)

    det = AnomalyDetector().fit(Xtr)        # il n'utilise PAS les étiquettes d'accident
    det.save()
    out = det.predict(Xte)

    flagged = out["is_anomaly"].to_numpy()
    print(f"Détecteur sauvegardé -> {ANOMALY_PATH}")
    print(f"Heures signalées anormales sur le test : {flagged.mean()*100:.1f} %"
          f"   |   porte ouverte (envoyées au modèle) : {out['gate_open'].mean()*100:.1f} %")
    print("\nVérification (le détecteur n'a jamais vu d'accident) :")
    print(f"  taux d'accident parmi les heures anormales : {100*yte[flagged].mean():.1f} %"
          f"   |   parmi les autres : {100*yte[~flagged].mean():.1f} %")
    print(f"  ROC-AUC du score contre les accidents : {roc_auc_score(yte, out['score']):.3f}"
          f"   PR-AUC : {average_precision_score(yte, out['score']):.3f}"
          f"   (hasard : {yte.mean():.3f})")

    print("\nExemples d'alertes (les 5 plus fortes) :")
    top = out.sort_values("score", ascending=False).head(5)
    for idx, row in top.iterrows():
        print(f"  {mte.loc[idx, 'intersection_id']} {mte.loc[idx, 'timestamp']:%d/%m %Hh} "
              f"score={row['score']:.2f}  {row['reasons'] or '(étrangeté globale)'}")