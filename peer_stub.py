"""
peer_stub.py : FAUX nœud voisin, pour tester le protocole AVANT que tes collègues aient fini les leurs.

Il parle exactement la même langue qu'un vrai nœud (mêmes points d'accès, même message NodeSummary),
mais son niveau d'alerte est imposé par celui qui l'appelle :

    python peer_stub.py --node-id fog_node_1 --zone-id 1 --port 8001 \
        --peers "fog_node_2=http://localhost:8002,fog_node_3=http://localhost:8003"

    curl -X POST localhost:8001/tick -H "Content-Type: application/json" -d '{"level": "alert", "risk": 0.9, "confirm": "reject"}'

Attention : ce n'est PAS une vraie zone. Il sert à vérifier la communication et la décision collaborative.
"""
from __future__ import annotations

import argparse

import pandas as pd

from api import create_app
from collaboration import AlertEvent, NodeSummary, Verdict


class ScriptedEngine:
    def __init__(self, node_id: str, zone_id: int, zone_name: str = "stub", start="2026-03-02 05:00"):
        self.node_id, self.zone_id, self.zone_name = node_id, zone_id, zone_name
        self._t = pd.Timestamp(start) - pd.Timedelta(hours=1)
        self.n = 8
        self.confirm_mode = "confirm"       # verdict donné aux alertes reçues : confirm / reject / unable

    @property
    def data_time(self):
        return self._t.to_pydatetime()

    def seek(self, ts) -> None:
        self._t = pd.Timestamp(ts) - pd.Timedelta(hours=1)

    def tick(self, level: str = "normal", risk: float | None = None, confirm: str = "confirm") -> NodeSummary:
        self.confirm_mode = confirm
        self._t += pd.Timedelta(hours=1)
        default = {"normal": 0.1, "surveillance": 0.25, "alert": 0.9}
        return NodeSummary(
            node_id=self.node_id, zone_id=self.zone_id, zone_name=self.zone_name,
            data_time=self._t.to_pydatetime(), alert_level=level,
            risk_score=default[level] if risk is None else risk, n_intersections=self.n,
            intersections_in_alert=[f"INT_{self.zone_id}01"] if level == "alert" else [],
        )

    def build_event(self, summary: NodeSummary):
        """Alerte factice : SANS mesures (un vrai nœud doit en joindre, sinon ses voisins répondent 'unable')."""
        if summary.alert_level != "alert":
            return None
        return AlertEvent(event_id=f"{self.node_id}-{self._t:%m%d%H}", node_id=self.node_id, zone_id=self.zone_id,
                          zone_name=self.zone_name, data_time=summary.data_time, kind="accident_risk",
                          risk_score=summary.risk_score, intersections=summary.intersections_in_alert)

    def confirm(self, event: AlertEvent) -> Verdict:
        return Verdict(event_id=event.event_id, node_id=self.node_id, verdict=self.confirm_mode,
                       own_risk=0.8 if self.confirm_mode == "confirm" else 0.1,
                       note="verdict scénarisé (faux nœud)")


def make_app(node_id: str, zone_id: int, peers: dict[str, str] | None = None, start="2026-03-02 05:00"):
    return create_app(engine=ScriptedEngine(node_id, zone_id, start=start), peers=peers, scripted=True)


if __name__ == "__main__":
    import uvicorn
    from collaboration import peers_from_env

    ap = argparse.ArgumentParser(description="Faux nœud fog voisin (tests)")
    ap.add_argument("--node-id", required=True)
    ap.add_argument("--zone-id", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--peers", default="", help='"nom=url,nom=url" (sinon FOG_PEERS / config.PEERS)')
    ap.add_argument("--start", default="2026-03-02 05:00")
    args = ap.parse_args()

    if args.peers:
        peers = dict(p.split("=", 1) for p in args.peers.split(",") if "=" in p)
    else:
        peers = peers_from_env()
    uvicorn.run(make_app(args.node_id, args.zone_id, peers, args.start), host="127.0.0.1",
                port=args.port, log_level="warning")