"""
demo_cluster.py : lance 3 nœuds en local et montre la décision collaborative.

    fog_node_2 (port 8002) = TON vrai nœud : données de la zone 2, vrais modèles
    fog_node_1 (8001) et fog_node_3 (8003) = faux voisins (peer_stub.py) dont le niveau est scénarisé

Prérequis :  python model.py  puis  python anomaly.py   (crée les fichiers .joblib)
Lancement :  python demo_cluster.py
Les journaux des serveurs sont écrits dans logs_<nom>.txt.
"""
from __future__ import annotations

import subprocess
import sys
import time

import httpx

PORTS = {"fog_node_1": 8001, "fog_node_2": 8002, "fog_node_3": 8003}
URL = {n: f"http://127.0.0.1:{p}" for n, p in PORTS.items()}
START = "2026-03-02T05:00:00"
TTL = "3"            # secondes : très court pour que la démo de panne soit rapide (120 par défaut)

# Niveaux imposés aux faux voisins, heure par heure (05h ... 10h), et leur AVIS sur les alertes de la zone 2.
# La zone 2 (réelle) fait ce que disent ses données.
#          heure  zone 1          zone 3     avis zone 1  avis zone 3
SCRIPT = [
    ("05h", "normal",       "normal", "confirm", "confirm"),
    ("06h", "normal",       "normal", "confirm", "confirm"),
    ("07h", "surveillance", "normal", "confirm", "confirm"),
    ("08h", "alert",        "normal", "confirm", "confirm"),   # alerte réelle en zone 2 : les 2 voisins confirment
    ("09h", "normal",       "alert",  "reject",  "confirm"),   # alerte réelle en zone 2 : zone 1 n'est pas d'accord
    ("10h", "normal",       "normal", "confirm", "confirm"),
]


def spawn(name: str, cmd: list[str], env_peers: str):
    import os
    env = {**os.environ, "FOG_PEERS": env_peers, "FOG_TTL_S": TTL, "PYTHONIOENCODING": "utf-8"}
    log = open(f"logs_{name}.txt", "w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log, stderr=log, env=env)


def wait_ready(client: httpx.Client, name: str, proc, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} s'est arrêté : voir logs_{name}.txt")
        try:
            if client.get(f"{URL[name]}/health").status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"{name} ne répond pas (voir logs_{name}.txt)")


def main() -> None:
    peers = lambda me: ",".join(f"{n}={u}" for n, u in URL.items() if n != me)
    procs = {}
    try:
        for name, port in PORTS.items():
            if name == "fog_node_2":
                cmd = [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1",
                       "--port", str(port), "--log-level", "warning"]
            else:
                cmd = [sys.executable, "peer_stub.py", "--node-id", name, "--zone-id", name[-1],
                       "--port", str(port)]
            procs[name] = spawn(name, cmd, peers(name))

        with httpx.Client(timeout=10) as c:
            for name in PORTS:
                wait_ready(c, name, procs[name])
            print("3 nœuds démarrés.\n")
            for name in PORTS:
                c.post(f"{URL[name]}/seek", json={"to": START})

            print("Heure | zone 1       | zone 2 (réelle)           | zone 3       | décision régionale vue par fog_node_2")
            print("------+--------------+---------------------------+--------------+---------------------------------------")
            for label, l1, l3, c1, c3 in SCRIPT:
                c.post(f"{URL['fog_node_1']}/tick", json={"level": l1, "confirm": c1})
                c.post(f"{URL['fog_node_3']}/tick", json={"level": l3, "confirm": c3})
                r = c.post(f"{URL['fog_node_2']}/tick").json()
                own = r["summary"]
                d = c.get(f"{URL['fog_node_2']}/decision").json()
                real = f"{own['alert_level']} (risque {own['risk_score']:.2f})"
                print(f"{label:5s} | {l1:12s} | {real:25s} | {l3:12s} | {d['level']:12s} {d['reason']}")
                if r["alert"]:
                    a = r["alert"]
                    avis = ", ".join(f"{n}: {v['verdict']}" for n, v in a["verdicts"].items())
                    print(f"      >>> ALERTE IMMÉDIATE de la zone 2 ({a['event']['kind']}, {a['event']['intersections']}) "
                          f"- réponses en {a['answered_in_ms']} ms - {avis}")
                    print(f"      >>> statut : {a['status']['status']} - {a['status']['reason']}")

            print("\n--- Panne : fog_node_3 s'arrête ---")
            procs["fog_node_3"].terminate()
            procs["fog_node_3"].wait(timeout=10)
            time.sleep(float(TTL) + 1)        # on laisse expirer son dernier résumé
            c.post(f"{URL['fog_node_1']}/tick", json={"level": "normal", "confirm": "reject"})
            r = c.post(f"{URL['fog_node_2']}/tick").json()
            d = c.get(f"{URL['fog_node_2']}/decision").json()
            print(f"11h   | zone 2 (réelle) : {r['summary']['alert_level']} (risque {r['summary']['risk_score']:.2f})")
            if r["alert"]:
                a = r["alert"]
                print("      alerte immédiate : " + ", ".join(f"{n}: {v['verdict']}" for n, v in a["verdicts"].items())
                      + f" -> {a['status']['status']} ({a['status']['reason']})")
            print(f"envoi des résumés : {r['delivery']}")
            print(f"décision : {d['level']} | dégradée = {d['degraded']} | injoignables = {d['missing']}")
            print(f"raison : {d['reason']}")
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()