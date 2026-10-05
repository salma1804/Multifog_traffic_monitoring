"""
node_engine.py : le "cerveau local" du nœud. À chaque tick (= une heure de données qui arrive) :
    1. récupère les mesures de l'heure pour les intersections de la zone
    2. les features viennent de preprocessing.py (historique propre au nœud)
    3. model.py donne le risque d'accident, anomaly.py repère le trafic anormal
    4. le tout est résumé en un NodeSummary (le message envoyé aux autres nœuds)

Les données viennent du CSV rejoué heure par heure (période de test), comme dans simulator.py.
"""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from anomaly import AnomalyDetector
from collaboration import AlertEvent, NodeSummary, Verdict
from config import NODE_CONFIG, TRAIN_END
from model import RiskModel
from preprocessing import build_features, load_data


class EndOfData(Exception):
    pass


def analyse(model: RiskModel, detector: AnomalyDetector, Xh: pd.DataFrame) -> pd.DataFrame:
    """LA CASCADE. Le gardien (anomaly) regarde toutes les lignes ; le médecin (modèle) n'est appelé QUE sur
    celles où la porte est ouverte. Résultat, une étiquette par intersection :
        critical_accident : le modèle confirme le risque d'accident            -> alerte immédiate
        congestion        : trafic anormal mais le modèle ne voit pas d'accident -> à surveiller, pas d'alerte
        watch             : le modèle est inquiet sans passer le seuil          -> à surveiller
        normal            : rien d'anormal (le modèle n'a même pas été lancé)"""
    anom = detector.predict(Xh)
    out = pd.DataFrame({"risk": 0.0, "alert": False, "model_ran": anom["gate_open"].to_numpy(),
                        "is_anomaly": anom["is_anomaly"].to_numpy(), "score": anom["score"].to_numpy(),
                        "reasons": anom["reasons"].to_numpy()}, index=Xh.index)
    gate = anom["gate_open"].to_numpy()
    watch = np.zeros(len(Xh), dtype=bool)
    if gate.any():
        r = model.predict(Xh[gate])
        out.loc[gate, "risk"] = r["risk"].to_numpy()
        out.loc[gate, "alert"] = r["alert"].to_numpy()
        watch[gate] = (r["level"] == "SURVEILLANCE").to_numpy()
    out["status"] = np.select(
        [out["alert"], out["is_anomaly"], watch], ["critical_accident", "congestion", "watch"], "normal")
    return out


class NodeEngine:
    def __init__(self):
        X, _, meta = build_features(load_data())
        mask = (meta["timestamp"] >= pd.Timestamp(TRAIN_END)).to_numpy()
        self.X = X[mask].reset_index(drop=True)
        self.meta = meta[mask].reset_index(drop=True)
        self.hours = sorted(self.meta["timestamp"].unique())
        self.pos = -1                                   # index de la dernière heure traitée
        self.model = RiskModel()                        # lance d'abord `python model.py` si absent
        self.detector = AnomalyDetector.load()          # idem `python anomaly.py`

    @property
    def data_time(self):
        return None if self.pos < 0 else pd.Timestamp(self.hours[self.pos]).to_pydatetime()

    def seek(self, ts) -> None:
        """Se place juste AVANT l'heure `ts` : le prochain tick traitera `ts`."""
        ts = pd.Timestamp(ts)
        hours = pd.DatetimeIndex(self.hours)
        if ts not in hours:
            raise ValueError(f"heure absente des données : {ts}")
        self.pos = int(hours.get_loc(ts)) - 1

    def tick(self) -> NodeSummary:
        if self.pos + 1 >= len(self.hours):
            raise EndOfData("fin du jeu de données")
        self.pos += 1
        ts = pd.Timestamp(self.hours[self.pos])
        idx = np.where((self.meta["timestamp"] == ts).to_numpy())[0]
        Xh = self.X.iloc[idx]
        inters = self.meta["intersection_id"].iloc[idx].tolist()

        a = analyse(self.model, self.detector, Xh)
        self._last = {"idx": idx, "inters": inters, "analysis": a}
        st = a["status"].to_numpy()
        in_alert = [i for i, v in zip(inters, st) if v == "critical_accident"]
        anomalous = [i for i, v in zip(inters, st) if v == "congestion"]     # = embouteillages à surveiller

        if in_alert:
            level = "alert"
        elif anomalous or (st == "watch").any():
            level = "surveillance"
        else:
            level = "normal"

        return NodeSummary(
            node_id=NODE_CONFIG["node_id"], zone_id=NODE_CONFIG["zone_id"],
            zone_name=NODE_CONFIG["zone_name"], data_time=ts.to_pydatetime(),
            alert_level=level, risk_score=float(a["risk"].max()), n_intersections=len(idx),
            intersections_in_alert=in_alert, intersections_anomalous=anomalous,
        )

    # ------------------------------------------------------------------ #
    # Alerte immédiate : construire le message, et juger celui d'un voisin
    # ------------------------------------------------------------------ #
    def build_event(self, summary: NodeSummary) -> AlertEvent | None:
        """Message d'alerte à envoyer tout de suite aux voisins (None s'il n'y a rien d'anormal)."""
        if summary.intersections_in_alert:
            kind, inters = "accident_risk", summary.intersections_in_alert
        elif summary.intersections_anomalous:
            kind, inters = "congestion", summary.intersections_anomalous
        else:
            return None
        last = self._last
        pos = {i: k for k, i in enumerate(last["inters"])}
        Xh = self.X.iloc[last["idx"]]
        # mesures de chaque intersection concernée ; pas de NaN (le JSON n'en accepte pas)
        clean = pd.DataFrame(np.nan_to_num(Xh.to_numpy(dtype=float)), columns=Xh.columns)
        feats = {i: {c: float(v) for c, v in clean.iloc[pos[i]].items()} for i in inters}
        a = last["analysis"]
        reasons = [f"{i} : {a['reasons'].iloc[pos[i]]}" for i in inters if a["reasons"].iloc[pos[i]]]
        risk = float(max(a["risk"].iloc[pos[i]] for i in inters))
        return AlertEvent(event_id=uuid.uuid4().hex[:12], node_id=summary.node_id, zone_id=summary.zone_id,
                          zone_name=summary.zone_name, data_time=summary.data_time, kind=kind,
                          risk_score=risk, intersections=inters, reasons=reasons, features=feats)

    def confirm(self, event: AlertEvent) -> Verdict:
        """Deuxième avis : on réévalue les MESURES reçues avec NOTRE modèle (entraîné sur NOTRE zone)."""
        me = NODE_CONFIG["node_id"]
        if not event.features:
            return Verdict(event_id=event.event_id, node_id=me, verdict="unable",
                           note="l'alerte ne contient pas de mesures")
        df = pd.DataFrame.from_dict(event.features, orient="index")
        needed = set(self.model.columns) if event.kind == "accident_risk" else set(self.detector.columns)
        missing = needed - set(df.columns)
        if missing and len(missing) > 0.1 * len(needed):
            return Verdict(event_id=event.event_id, node_id=me, verdict="unable",
                           note=f"{len(missing)} colonnes manquantes (preprocessing différent ?)")
        if event.kind == "accident_risk":
            r = self.model.predict(df)
            ok, own = bool(r["alert"].any()), float(r["risk"].max())
            note = f"mon modèle donne un risque de {own:.2f} (seuil {self.model.threshold:.2f})"
        else:
            a = self.detector.predict(df)
            ok, own = bool(a["is_anomaly"].any()), None
            note = f"mon détecteur {'signale' if ok else 'ne signale pas'} ces mesures comme anormales"
        return Verdict(event_id=event.event_id, node_id=me, verdict="confirm" if ok else "reject",
                       own_risk=own, note=note)


if __name__ == "__main__":
    # Évalue la cascade sur le jeu de test, face au modèle seul.
    from sklearn.metrics import confusion_matrix  # noqa: F401
    from preprocessing import temporal_split

    X, y, meta = build_features(load_data())
    _, Xte, _, yte, _, _ = temporal_split(X, y, meta)
    model, det = RiskModel(), AnomalyDetector.load()
    a = analyse(model, det, Xte)
    yv = np.asarray(yte).astype(bool)

    def line(name, alert):
        tp, fp = int((alert & yv).sum()), int((alert & ~yv).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        print(f"{name:34s} accidents trouvés {tp:3d}/{int(yv.sum())}   fausses alertes {fp:3d}   précision {prec:.0%}")

    alone = model.predict(Xte)["alert"].to_numpy()
    print(f"Modèle appelé sur {a['model_ran'].mean():.0%} des lignes (au lieu de 100 %)\n")
    line("modèle seul (sur tout)", alone)
    line("cascade (porte large)", a["alert"].to_numpy())
    print("\nÉtiquettes de la cascade sur le test :")
    tab = pd.crosstab(a["status"], pd.Series(np.where(yv, "accident réel", "pas d'accident"), index=a.index))
    print(tab.to_string())