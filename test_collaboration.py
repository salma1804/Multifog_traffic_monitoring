"""
Tests automatiques :  pytest -q
Ils ne lancent aucun serveur : l'API est appelée en mémoire (TestClient).
"""
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import create_app
import api as api_module
from collaboration import AlertEvent, NodeSummary, Received, Verdict, decide, resolve_event
from peer_stub import ScriptedEngine

T = datetime(2026, 3, 2, 8, 0)
PEERS = {"fog_node_1": "http://x", "fog_node_2": "http://x", "fog_node_3": "http://x"}


def summ(node_id, zone_id, level, risk=0.1, t=T):
    return NodeSummary(node_id=node_id, zone_id=zone_id, data_time=t, alert_level=level,
                       risk_score=risk, n_intersections=8)


def received(*items, at=1000.0):
    return {s.node_id: Received(s, at) for s in items}


# ------------------------- la règle de décision ------------------------------ #
EXPECTED = ["fog_node_1", "fog_node_2", "fog_node_3"]


def test_all_normal():
    d = decide(summ("fog_node_2", 2, "normal"), received(summ("fog_node_1", 1, "normal"), summ("fog_node_3", 3, "normal")),
               EXPECTED, now=1010)
    assert d.level == "normal" and not d.degraded and d.zones_reporting == 3


def test_one_zone_alert_is_watch():
    d = decide(summ("fog_node_2", 2, "alert", 0.9), received(summ("fog_node_1", 1, "normal"), summ("fog_node_3", 3, "normal")),
               EXPECTED, now=1010)
    assert d.level == "watch" and "fog_node_2" in d.reason


def test_two_alerts_is_global():
    d = decide(summ("fog_node_2", 2, "alert"), received(summ("fog_node_1", 1, "alert"), summ("fog_node_3", 3, "normal")),
               EXPECTED, now=1010)
    assert d.level == "global_alert"


def test_two_surveillance_is_watch():
    d = decide(summ("fog_node_2", 2, "surveillance"), received(summ("fog_node_1", 1, "surveillance"), summ("fog_node_3", 3, "normal")),
               EXPECTED, now=1010)
    assert d.level == "watch"


def test_stale_peer_is_ignored_and_flagged():
    old = received(summ("fog_node_1", 1, "alert"), summ("fog_node_3", 3, "normal"), at=1000.0)
    d = decide(summ("fog_node_2", 2, "normal"), old, EXPECTED, now=1000.0 + 500, ttl_s=120)
    assert d.degraded and sorted(d.missing) == ["fog_node_1", "fog_node_3"]
    assert d.level == "normal"                      # l'alerte périmée du voisin n'est pas utilisée


def test_lone_alert_with_unreachable_peers_is_never_normal():
    d = decide(summ("fog_node_2", 2, "alert", 0.9), {}, EXPECTED, now=1000.0)
    assert d.level == "watch" and d.degraded and "injoignables" in d.reason


def test_misaligned_hours_are_flagged():
    other = datetime(2026, 3, 2, 9, 0)
    d = decide(summ("fog_node_2", 2, "normal"), received(summ("fog_node_1", 1, "normal", t=other), summ("fog_node_3", 3, "normal")),
               EXPECTED, now=1010)
    assert not d.aligned


# ------------------------------- l'API --------------------------------------- #
@pytest.fixture
def client():
    # push=False : aucun envoi réseau pendant les tests ; `with` exécute le démarrage de l'application
    app = create_app(engine=ScriptedEngine("fog_node_2", 2), peers=PEERS, push=False, scripted=True)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["node_id"] == "fog_node_2" and r["status"] == "ok"
    assert "fog_node_2" not in r["peers"]            # un nœud n'est pas son propre voisin


def test_decision_requires_a_tick_first(client):
    assert client.get("/decision").status_code == 409
    assert client.get("/summary").status_code == 409


def test_full_flow_two_alerts_make_global_alert(client):
    client.post("/tick", json={"level": "alert", "risk": 0.9})
    client.post("/peer/summary", json=summ("fog_node_1", 1, "alert", 0.8).model_dump(mode="json"))
    client.post("/peer/summary", json=summ("fog_node_3", 3, "normal").model_dump(mode="json"))
    d = client.get("/decision").json()
    assert d["level"] == "global_alert" and d["zones_reporting"] == 3 and not d["degraded"]


def test_rejects_wrong_schema_version(client):
    bad = summ("fog_node_1", 1, "normal").model_dump(mode="json") | {"schema_version": 99}
    assert client.post("/peer/summary", json=bad).status_code == 422


def test_rejects_invalid_risk_score(client):
    bad = summ("fog_node_1", 1, "normal").model_dump(mode="json") | {"risk_score": 3.5}
    assert client.post("/peer/summary", json=bad).status_code == 422


def test_rejects_unknown_node_and_own_messages(client):
    assert client.post("/peer/summary", json=summ("fog_node_9", 9, "normal").model_dump(mode="json")).status_code == 422
    assert client.post("/peer/summary", json=summ("fog_node_2", 2, "normal").model_dump(mode="json")).status_code == 422


def test_unreachable_peer_does_not_break_tick():
    app = create_app(engine=ScriptedEngine("fog_node_2", 2), peers={"fog_node_1": "http://127.0.0.1:9"}, push=True, scripted=True)
    with TestClient(app) as c:
        r = c.post("/tick", json={"level": "normal"})
        assert r.status_code == 200 and r.json()["delivery"]["fog_node_1"]["ok"] is False


# ----------------------- alerte immédiate et confirmation -------------------- #
def verdict(node, v):
    return Verdict(event_id="e1", node_id=node, verdict=v)


def test_no_neighbor_can_judge_is_unverified():
    assert resolve_event({}).status == "unverified"
    assert resolve_event({"a": verdict("a", "unable"), "b": verdict("b", "unreachable")}).status == "unverified"


def test_one_confirmation_is_enough_with_three_nodes():
    st = resolve_event({"a": verdict("a", "confirm"), "b": verdict("b", "reject")})
    assert st.status == "confirmed" and st.confirmed_by == ["a"] and st.rejected_by == ["b"]
    assert resolve_event({"a": verdict("a", "confirm"), "b": verdict("b", "unreachable")}).status == "confirmed"


def test_rejection_by_neighbors_means_unconfirmed():
    assert resolve_event({"a": verdict("a", "reject"), "b": verdict("b", "reject")}).status == "unconfirmed"
    st = resolve_event({"a": verdict("a", "reject"), "b": verdict("b", "unreachable")})
    assert st.status == "unconfirmed" and st.no_opinion == ["b"]


def alert_event(**kw):
    base = dict(event_id="e1", node_id="fog_node_1", zone_id=1, data_time=T, kind="accident_risk",
                risk_score=0.9, intersections=["INT_001"])
    return AlertEvent(**{**base, **kw})


def test_api_peer_alert_returns_a_verdict_immediately(client):
    client.post("/tick", json={"level": "normal", "confirm": "reject"})
    r = client.post("/peer/alert", json=alert_event().model_dump(mode="json"))
    assert r.status_code == 200 and r.json()["verdict"] == "reject" and r.json()["node_id"] == "fog_node_2"
    assert client.get("/events").json()["received"][0]["my_verdict"]["verdict"] == "reject"


def test_api_peer_alert_rejects_unknown_sender_and_bad_version(client):
    assert client.post("/peer/alert", json=alert_event(node_id="fog_node_9").model_dump(mode="json")).status_code == 422
    assert client.post("/peer/alert", json=alert_event().model_dump(mode="json") | {"schema_version": 5}).status_code == 422


def test_tick_with_alert_sends_event_first_then_summary(monkeypatch):
    calls = []
    monkeypatch.setattr(api_module, "send_alert", lambda ev, peers: calls.append("alert") or
                        {"fog_node_1": Verdict(event_id=ev.event_id, node_id="fog_node_1", verdict="confirm"),
                         "fog_node_3": Verdict(event_id=ev.event_id, node_id="fog_node_3", verdict="reject")})
    monkeypatch.setattr(api_module, "broadcast", lambda s, peers: calls.append("summary") or {})
    app = create_app(engine=ScriptedEngine("fog_node_2", 2), peers=PEERS, push=True, scripted=True)
    with TestClient(app) as c:
        r = c.post("/tick", json={"level": "alert", "risk": 0.9}).json()
        assert calls == ["alert", "summary"]                       # l'alerte part AVANT le résumé
        assert r["alert"]["status"]["status"] == "confirmed"
        assert c.get("/events").json()["emitted"][0]["event"]["node_id"] == "fog_node_2"


def test_tick_without_anything_abnormal_sends_no_alert(client):
    assert client.post("/tick", json={"level": "normal"}).json()["alert"] is None


# ------------------- le vrai moteur (si modèles et CSV présents) -------------- #
needs_artifacts = pytest.mark.skipif(
    not (Path("model_fog_node_2.joblib").exists() and Path("anomaly_fog_node_2.joblib").exists()),
    reason="lance d'abord  python model.py  et  python anomaly.py")


@needs_artifacts
def test_real_engine_raises_alert_on_known_accident_hour():
    from node_engine import NodeEngine
    e = NodeEngine()
    e.seek("2026-03-02 08:00")
    s = e.tick()
    assert s.alert_level == "alert" and s.intersections_in_alert and s.risk_score > 0.5


@needs_artifacts
def test_real_engine_is_quiet_at_night():
    from node_engine import NodeEngine
    e = NodeEngine()
    e.seek("2026-03-02 03:00")
    assert e.tick().alert_level in ("normal", "surveillance")


@needs_artifacts
def test_real_alert_carries_measurements_and_neighbor_with_same_model_confirms():
    from node_engine import NodeEngine
    e = NodeEngine()
    e.seek("2026-03-02 08:00")
    ev = e.build_event(e.tick())
    assert ev is not None and ev.kind == "accident_risk" and ev.intersections
    feats = ev.features[ev.intersections[0]]
    assert len(feats) > 40 and all(isinstance(v, float) for v in feats.values())
    AlertEvent.model_validate_json(ev.model_dump_json())             # le message passe bien en JSON
    v = e.confirm(ev)                                                # même modèle, mêmes mesures
    assert v.verdict == "confirm" and v.own_risk > 0.5


@needs_artifacts
def test_real_neighbor_rejects_calm_measurements_and_cannot_judge_without_them():
    from node_engine import NodeEngine
    e = NodeEngine()
    e.seek("2026-03-02 08:00")
    ev = e.build_event(e.tick())
    e.seek("2026-03-02 03:00")
    e.tick()
    calm = e._last
    Xh = e.X.iloc[calm["idx"]]
    calm_feats = {calm["inters"][0]: {c: float(v) for c, v in Xh.iloc[0].items()}}
    assert e.confirm(ev.model_copy(update={"features": calm_feats})).verdict == "reject"
    assert e.confirm(ev.model_copy(update={"features": {}})).verdict == "unable"
    truncated = {k: dict(list(v.items())[:10]) for k, v in ev.features.items()}
    assert e.confirm(ev.model_copy(update={"features": truncated})).verdict == "unable"


# ---------------------- la cascade : gardien (anomaly) puis médecin (modèle) -------- #
@needs_artifacts
def test_cascade_runs_model_only_when_gate_is_open_and_labels_are_consistent():
    from node_engine import NodeEngine, analyse
    e = NodeEngine()
    a = analyse(e.model, e.detector, e.X.iloc[:600])
    assert 0.1 < a["model_ran"].mean() < 0.6                              # le modèle ne tourne pas sur tout
    assert (a.loc[~a["model_ran"], "risk"] == 0).all()                    # porte fermée -> modèle jamais appelé
    assert (a.loc[a["status"] == "critical_accident", "model_ran"]).all() # une alerte vient toujours de la porte ouverte
    assert (a.loc[a["status"] == "congestion", "alert"] == False).all()   # congestion = le modèle n'a PAS d'accident


@needs_artifacts
def test_cascade_keeps_every_true_accident_the_model_alone_finds_and_drops_false_alerts():
    import numpy as np
    from config import TRAIN_END
    from node_engine import NodeEngine, analyse
    from preprocessing import build_features, load_data
    _, y, meta = build_features(load_data())
    yv = np.asarray(y)[(meta["timestamp"] >= pd.Timestamp(TRAIN_END)).to_numpy()].astype(bool)
    e = NodeEngine()
    cascade = analyse(e.model, e.detector, e.X)["alert"].to_numpy()
    alone = e.model.predict(e.X)["alert"].to_numpy()
    assert not (cascade & ~alone).any()                                   # la cascade n'invente aucune alerte
    assert (cascade & yv).sum() == (alone & yv).sum()                     # vrais accidents : aucun perdu
    assert (cascade & ~yv).sum() < (alone & ~yv).sum()                    # fausses alertes : moins nombreuses


@needs_artifacts
def test_old_detector_file_without_gate_is_refused(tmp_path):
    import joblib
    from anomaly import AnomalyDetector
    joblib.dump({"params": {}, "columns": [], "iso": None, "threshold": 0.0, "rules": {}}, tmp_path / "old.joblib")
    with pytest.raises(ValueError, match="python anomaly.py"):
        AnomalyDetector.load(str(tmp_path / "old.joblib"))


def test_congestion_event_is_accepted_by_the_contract_and_gets_a_vote(client):
    ev = alert_event(kind="congestion", risk_score=0.1)
    assert AlertEvent.model_validate_json(ev.model_dump_json()).kind == "congestion"
    client.post("/tick", json={"level": "normal", "confirm": "confirm"})
    assert client.post("/peer/alert", json=ev.model_dump(mode="json")).json()["verdict"] == "confirm"