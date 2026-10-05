"""
api.py : l'interface réseau du nœud fog (FastAPI). C'est ce qui fait de ton projet un VRAI nœud :
un service qui tourne, que les autres nœuds (et DevOps) peuvent interroger.

Lancement :
    uvicorn api:app --host 0.0.0.0 --port 8002
Voisins (au choix) :
    - variable d'environnement  FOG_PEERS="fog_node_1=http://localhost:8001,fog_node_3=http://localhost:8003"
    - ou  PEERS = {...}  dans config.py
Durée de validité d'un résumé reçu :  FOG_TTL_S  (secondes, 120 par défaut)

Points d'accès :
    GET  /health          le nœud est-il vivant ? (pour Docker / supervision)
    GET  /summary         le dernier résumé de CE nœud
    POST /tick            traite l'heure suivante. Si quelque chose est détecté : ALERTE IMMÉDIATE aux voisins
                          (qui répondent par un verdict), PUIS résumé périodique
    POST /peer/alert      reçoit l'alerte d'un voisin, la réévalue avec NOTRE modèle, répond par un verdict
    GET  /events          alertes émises (avec verdicts et statut) et alertes reçues
    POST /seek            (rejeu) se placer juste avant une heure : {"to": "2026-03-02T05:00:00"}
    POST /peer/summary    reçoit le résumé d'un voisin  (appelé par les autres nœuds)
    GET  /peers           les résumés reçus et leur ancienneté
    GET  /decision        la décision collaborative (normal / watch / global_alert)
Documentation interactive automatique :  http://localhost:8002/docs

Attention : aucune authentification pour l'instant. À mettre derrière un réseau privé / reverse proxy.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel

from collaboration import (SCHEMA_VERSION, AlertEvent, Decision, NodeSummary, PeerStore, Verdict, broadcast,
                           decide, peers_from_env, resolve_event, send_alert)

log = logging.getLogger("fog.api")


def _trim(d: dict, keep: int = 200) -> None:
    """Garde seulement les `keep` entrées les plus récentes (la mémoire ne grossit pas sans fin)."""
    while len(d) > keep:
        d.pop(next(iter(d)))


class SeekRequest(BaseModel):
    to: datetime


def create_app(engine=None, peers: Optional[dict[str, str]] = None, ttl_s: Optional[float] = None,
               push: bool = True, scripted: bool = False) -> FastAPI:
    """Fabrique l'application. `engine`/`peers` peuvent être injectés (tests, nœuds factices)."""
    store = PeerStore()
    state = {"engine": engine, "own": None, "events": {}, "received": {}}
    ttl = float(os.environ.get("FOG_TTL_S", 120)) if ttl_s is None else ttl_s

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["engine"] is None:
            from node_engine import NodeEngine
            state["engine"] = NodeEngine()
        node_id = state["engine"].node_id if hasattr(state["engine"], "node_id") else None
        if node_id is None:
            from config import NODE_CONFIG
            node_id = NODE_CONFIG["node_id"]
        state["node_id"] = node_id
        wanted = dict(peers_from_env() if peers is None else peers)
        wanted.pop(node_id, None)                       # on ne s'envoie pas de message à soi-même
        state["peers"] = wanted
        log.info("nœud %s démarré, voisins : %s", node_id, list(wanted))
        yield

    app = FastAPI(title="Nœud fog — MultiFog Traffic Monitoring", lifespan=lifespan)

    def _engine():
        return state["engine"]

    @app.get("/health")
    def health():
        e = _engine()
        return {"node_id": state["node_id"], "status": "ok",
                "data_time": e.data_time, "peers": list(state["peers"])}

    @app.get("/summary", response_model=NodeSummary)
    def summary():
        if state["own"] is None:
            raise HTTPException(409, "aucun résumé : appelle d'abord POST /tick")
        return state["own"]

    @app.post("/tick")
    def tick(params: Optional[dict] = Body(default=None)):
        if params and not scripted:
            raise HTTPException(400, "ce nœud n'accepte pas de paramètres sur /tick")
        try:
            s = _engine().tick(**(params or {}))
        except Exception as e:
            if type(e).__name__ == "EndOfData":
                raise HTTPException(409, str(e))
            raise
        state["own"] = s

        # 1) ALERTE IMMÉDIATE : avant tout le reste, dès que quelque chose est détecté
        event_info = None
        ev = _engine().build_event(s) if hasattr(_engine(), "build_event") else None
        if ev is not None:
            t0 = time.perf_counter()
            verdicts = send_alert(ev, state["peers"]) if push else {}
            ms = round((time.perf_counter() - t0) * 1000, 1)
            st = resolve_event(verdicts)
            event_info = {"event": ev, "verdicts": verdicts, "status": st, "answered_in_ms": ms}
            state["events"][ev.event_id] = event_info
            _trim(state["events"])
            log.warning("ALERTE %s %s intersections=%s -> %s", ev.kind, ev.event_id, ev.intersections, st.status)

        # 2) résumé périodique
        delivery = broadcast(s, state["peers"]) if push else {}
        return {"summary": s, "alert": event_info, "delivery": delivery}

    @app.post("/seek")
    def seek(req: SeekRequest):
        try:
            _engine().seek(req.to)
        except ValueError as e:
            raise HTTPException(404, str(e))
        state["own"] = None
        return {"next_tick_processes": req.to}

    def _check_sender(schema_version: int, node_id: str) -> None:
        if schema_version != SCHEMA_VERSION:
            raise HTTPException(422, f"schema_version {schema_version} non supporté (attendu {SCHEMA_VERSION})")
        if node_id == state["node_id"]:
            raise HTTPException(422, "ce message vient de ce nœud lui-même")
        if state["peers"] and node_id not in state["peers"]:
            raise HTTPException(422, f"nœud inconnu '{node_id}', attendus : {sorted(state['peers'])}")

    @app.post("/peer/summary")
    def peer_summary(s: NodeSummary):
        _check_sender(s.schema_version, s.node_id)
        store.put(s)
        return {"ok": True}

    @app.post("/peer/alert", response_model=Verdict)
    def peer_alert(ev: AlertEvent):
        """Un voisin signale un problème : on réévalue SES mesures avec NOTRE modèle et on répond tout de suite."""
        _check_sender(ev.schema_version, ev.node_id)
        try:
            v = _engine().confirm(ev)
        except Exception as e:
            v = Verdict(event_id=ev.event_id, node_id=state["node_id"], verdict="unable",
                        note=f"erreur : {type(e).__name__}")
        state["received"][ev.event_id] = {"event": ev, "my_verdict": v}
        _trim(state["received"])
        log.warning("ALERTE REÇUE de %s (%s) -> mon verdict : %s", ev.node_id, ev.kind, v.verdict)
        return v

    @app.get("/events")
    def events_view():
        return {"emitted": list(state["events"].values()), "received": list(state["received"].values())}

    @app.get("/peers")
    def peers_view():
        now = time.time()
        return {nid: {"summary": r.summary, "age_s": round(now - r.received_at, 1),
                      "fresh": now - r.received_at <= ttl} for nid, r in store.snapshot().items()}

    @app.get("/decision", response_model=Decision)
    def decision():
        if state["own"] is None:
            raise HTTPException(409, "aucun résumé local : appelle d'abord POST /tick")
        return decide(state["own"], store.snapshot(), list(state["peers"]) + [state["node_id"]],
                      time.time(), ttl_s=ttl)

    return app


# `uvicorn api:app` : le moteur (modèles + données) est chargé au démarrage, pas à l'import.
app = create_app()