"""
collaboration.py : échange d'informations entre nœuds fog + décision collaborative.

PRINCIPE
--------
Chaque nœud traite SES alertes localement, sans attendre personne. Deux types de messages partent vers les voisins :

  1. ALERTE IMMÉDIATE (AlertEvent) : envoyée DÈS qu'un risque d'accident ou une anomalie est détecté, avec les
     mesures de l'intersection concernée. Chaque voisin les réévalue avec SON modèle et répond tout de suite par
     un verdict (Verdict : confirm / reject / unable). L'émetteur en déduit un statut : confirmed / unconfirmed /
     unverified. Cette confirmation n'est jamais attendue pour agir en local.
  2. RÉSUMÉ PÉRIODIQUE (NodeSummary) : l'état de la zone à chaque heure ("je suis vivant, voici mon niveau").
     Chacun garde le dernier résumé reçu de chaque voisin et en déduit une décision RÉGIONALE : "est-ce un
     problème isolé ou général ?"

    normal        : aucune zone en alerte
    watch         : une zone en alerte, ou au moins deux en surveillance (à surveiller)
    global_alert  : au moins deux zones en alerte (problème régional)

LE CONTRAT À RESPECTER PAR TOUS LES NŒUDS (voir NodeSummary) : mêmes noms de champs, mêmes valeurs.
Un résumé est envoyé en JSON à  POST  <url_du_voisin>/peer/summary .

PANNES : un voisin silencieux depuis plus de `ttl_s` secondes est considéré injoignable. La décision est
alors marquée `degraded` et un nœud seul en alerte n'est JAMAIS déclaré "normal" faute de confirmation.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, Field

SCHEMA_VERSION = 1
Level = Literal["normal", "surveillance", "alert"]
GlobalLevel = Literal["normal", "watch", "global_alert"]


# --------------------------------------------------------------------------- #
# 1) LE CONTRAT : le message échangé entre les nœuds
# --------------------------------------------------------------------------- #
class NodeSummary(BaseModel):
    schema_version: int = SCHEMA_VERSION
    node_id: str                                   # ex. "fog_node_2" (identique à la clé dans PEERS)
    zone_id: int
    zone_name: str = ""
    data_time: datetime                            # l'heure des mesures résumées (heure du dataset)
    alert_level: Level                             # niveau de la zone sur cette heure
    risk_score: float = Field(ge=0, le=1)          # risque maximal parmi les intersections
    n_intersections: int = Field(ge=0)
    intersections_in_alert: list[str] = []         # intersections dont le risque dépasse le seuil
    intersections_anomalous: list[str] = []        # intersections au trafic anormal
    load: Optional[float] = Field(default=None, ge=0, le=1)   # charge du nœud (0-1), pour l'offloading


# --------------------------------------------------------------------------- #
# 1 bis) ALERTE IMMÉDIATE et demande de confirmation
# --------------------------------------------------------------------------- #
EventKind = Literal["accident_risk", "congestion", "anomaly"]   # "anomaly" = ancien nom, toujours accepté
VerdictValue = Literal["confirm", "reject", "unable", "unreachable"]


class AlertEvent(BaseModel):
    """Envoyé tout de suite après une détection. Il contient les MESURES de l'intersection concernée, pour que
    les voisins puissent donner un deuxième avis. Tous les nœuds doivent utiliser les mêmes noms de colonnes
    (même preprocessing.py), sinon le voisin répond "unable"."""
    schema_version: int = SCHEMA_VERSION
    event_id: str
    node_id: str
    zone_id: int
    zone_name: str = ""
    data_time: datetime
    kind: EventKind                                 # accident_risk : accident critique (anomalie ET modèle d'accord) ; congestion : trafic anormal sans accident
    risk_score: float = Field(ge=0, le=1)
    intersections: list[str]
    reasons: list[str] = []
    features: dict[str, dict[str, float]] = {}      # intersection -> {colonne: valeur}


class Verdict(BaseModel):
    """Réponse d'un voisin. confirm : mon modèle voit aussi le problème ; reject : mon modèle ne le voit pas ;
    unable : je ne peux pas juger (mesures absentes, colonnes manquantes) ; unreachable : voisin injoignable."""
    event_id: str
    node_id: str
    verdict: VerdictValue
    own_risk: Optional[float] = None
    note: str = ""


class EventStatus(BaseModel):
    status: Literal["confirmed", "unconfirmed", "unverified"]
    reason: str
    confirmed_by: list[str] = []
    rejected_by: list[str] = []
    no_opinion: list[str] = []


def resolve_event(verdicts: dict[str, Verdict]) -> EventStatus:
    """Vote à la majorité entre les nœuds QUI ONT PU JUGER. L'émetteur vote pour sa propre alerte.
    3 nœuds : l'alerte est confirmée dès qu'au moins un voisin est d'accord (2 voix sur 3).
    Si aucun voisin n'a pu juger, l'alerte reste "unverified" : elle est quand même traitée en local."""
    yes = sorted(n for n, v in verdicts.items() if v.verdict == "confirm")
    no = sorted(n for n, v in verdicts.items() if v.verdict == "reject")
    none = sorted(n for n, v in verdicts.items() if v.verdict in ("unable", "unreachable"))
    if not yes and not no:
        reason = "aucun voisin n'a pu juger" + (f" ({', '.join(none)})" if none else " (aucun voisin)")
        return EventStatus(status="unverified", reason=reason, no_opinion=none)
    voters = 1 + len(yes) + len(no)
    if 2 * (1 + len(yes)) > voters:
        status, head = "confirmed", f"confirmée par {', '.join(yes) or '—'} ({1 + len(yes)} voix sur {voters})"
    else:
        status, head = "unconfirmed", f"non confirmée : rejetée par {', '.join(no)} ({1 + len(yes)} voix sur {voters})"
    if none:
        head += f" ; sans avis : {', '.join(none)}"
    return EventStatus(status=status, reason=head, confirmed_by=yes, rejected_by=no, no_opinion=none)


def _alert_one(name: str, url: str, payload: dict, event_id: str, timeout: float) -> tuple[str, Verdict]:
    try:
        r = httpx.post(f"{url.rstrip('/')}/peer/alert", json=payload, timeout=timeout)
        if r.status_code != 200:
            return name, Verdict(event_id=event_id, node_id=name, verdict="unable", note=f"HTTP {r.status_code}")
        v = Verdict.model_validate(r.json())
        return name, v.model_copy(update={"node_id": name})
    except httpx.HTTPError as e:
        return name, Verdict(event_id=event_id, node_id=name, verdict="unreachable", note=type(e).__name__)
    except ValueError:
        return name, Verdict(event_id=event_id, node_id=name, verdict="unable", note="réponse invalide")


def send_alert(event: AlertEvent, peers: dict[str, str], timeout: float = 2.0) -> dict[str, Verdict]:
    """Envoie l'alerte à tous les voisins EN PARALLÈLE et renvoie leurs verdicts (réponse immédiate)."""
    if not peers:
        return {}
    payload = event.model_dump(mode="json")
    with ThreadPoolExecutor(max_workers=len(peers)) as pool:
        futures = [pool.submit(_alert_one, n, u, payload, event.event_id, timeout) for n, u in peers.items()]
        return dict(f.result() for f in futures)


# --------------------------------------------------------------------------- #
# 2) Mémoire des résumés reçus
# --------------------------------------------------------------------------- #
@dataclass
class Received:
    summary: NodeSummary
    received_at: float          # heure de RÉCEPTION sur CE nœud (on ne se fie pas à l'horloge de l'envoyeur)


class PeerStore:
    """Dernier résumé reçu de chaque voisin (le plus récent écrase le précédent)."""

    def __init__(self):
        self._data: dict[str, Received] = {}
        self._lock = threading.Lock()

    def put(self, summary: NodeSummary, now: Optional[float] = None) -> None:
        with self._lock:
            self._data[summary.node_id] = Received(summary, time.time() if now is None else now)

    def snapshot(self) -> dict[str, Received]:
        with self._lock:
            return dict(self._data)


# --------------------------------------------------------------------------- #
# 3) La décision collaborative (fonction pure : facile à tester)
# --------------------------------------------------------------------------- #
class ZoneStatus(BaseModel):
    node_id: str
    zone_id: int
    level: Level
    risk_score: float
    data_time: datetime
    age_s: float                # ancienneté du résumé (0 pour soi-même)
    own: bool = False


class Decision(BaseModel):
    level: GlobalLevel
    reason: str
    zones_reporting: int
    zones_expected: int
    degraded: bool              # vrai si au moins un voisin est injoignable
    aligned: bool               # vrai si tous les résumés portent sur la même heure
    missing: list[str]
    zones: list[ZoneStatus]


def decide(own: NodeSummary, received: dict[str, Received], expected: list[str],
           now: float, ttl_s: float = 120.0, min_alert_zones: int = 2) -> Decision:
    expected_all = sorted(set(expected) | {own.node_id})
    zones = [ZoneStatus(node_id=own.node_id, zone_id=own.zone_id, level=own.alert_level,
                        risk_score=own.risk_score, data_time=own.data_time, age_s=0.0, own=True)]
    missing: list[str] = []
    for node_id in expected_all:
        if node_id == own.node_id:
            continue
        r = received.get(node_id)
        if r is None or now - r.received_at > ttl_s:
            missing.append(node_id)
            continue
        s = r.summary
        zones.append(ZoneStatus(node_id=s.node_id, zone_id=s.zone_id, level=s.alert_level,
                                risk_score=s.risk_score, data_time=s.data_time,
                                age_s=round(now - r.received_at, 1)))

    alerts = [z for z in zones if z.level == "alert"]
    survs = [z for z in zones if z.level == "surveillance"]
    names = lambda zs: ", ".join(z.node_id for z in zs)

    if len(alerts) >= min_alert_zones:
        level, reason = "global_alert", f"{len(alerts)} zones en alerte ({names(alerts)})"
    elif alerts:
        level, reason = "watch", f"1 zone en alerte ({names(alerts)})"
        if missing:
            reason += f" ; confirmation impossible, zones injoignables : {', '.join(missing)}"
    elif len(survs) >= 2:
        level, reason = "watch", f"{len(survs)} zones en surveillance ({names(survs)})"
    else:
        level, reason = "normal", "aucune zone en alerte"

    return Decision(level=level, reason=reason, zones_reporting=len(zones),
                    zones_expected=len(expected_all), degraded=bool(missing),
                    aligned=len({z.data_time for z in zones}) == 1, missing=missing, zones=zones)


# --------------------------------------------------------------------------- #
# 4) Envoi aux voisins
# --------------------------------------------------------------------------- #
def peers_from_env(env_var: str = "FOG_PEERS") -> dict[str, str]:
    """Voisins : variable d'environnement  FOG_PEERS="fog_node_1=http://hote:8001,fog_node_3=http://..."
    sinon la variable PEERS de config.py (optionnelle)."""
    raw = os.environ.get(env_var, "").strip()
    if raw:
        peers = {}
        for item in raw.split(","):
            name, _, url = item.strip().partition("=")
            if name.strip() and url.strip():
                peers[name.strip()] = url.strip()
        return peers
    try:
        import config
        return dict(getattr(config, "PEERS", {}))
    except ImportError:
        return {}


def _send_one(name: str, url: str, payload: dict, timeout: float) -> tuple[str, dict]:
    try:
        r = httpx.post(f"{url.rstrip('/')}/peer/summary", json=payload, timeout=timeout)
        return name, {"ok": r.status_code == 200, "status": r.status_code}
    except httpx.HTTPError as e:
        return name, {"ok": False, "error": type(e).__name__}


def broadcast(summary: NodeSummary, peers: dict[str, str], timeout: float = 2.0) -> dict[str, dict]:
    """Envoie le résumé à tous les voisins EN PARALLÈLE. Un voisin en panne ne bloque ni ne fait
    échouer les autres : on note simplement l'échec."""
    if not peers:
        return {}
    payload = summary.model_dump(mode="json")
    with ThreadPoolExecutor(max_workers=len(peers)) as pool:
        futures = [pool.submit(_send_one, n, u, payload, timeout) for n, u in peers.items()]
        return dict(f.result() for f in futures)