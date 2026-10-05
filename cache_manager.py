"""
cache_manager.py : kn khdemha khdmha 9bal may3wdhech snn kn makhdmhesh yaraha tchbh l ama haja khdmjha kbal

IDÉE
----
Si le nœud a DÉJÀ calculé le risque d'accident pour une situation quasi identique
(même intersection, trafic du même ordre de grandeur), inutile de tout recalculer :
il relit la réponse mémorisée. Ça coûte presque rien (0.05 au lieu de 5).

Trois modes comparés :
    no_cache        : on recalcule toujours
    lru             : cache RÉACTIF. Le 1er calcul est mémorisé, les suivants (situation similaire,
                      moins de 6 h plus tard) sont des "hits". LRU = quand c'est plein, on oublie
                      l'entrée la moins récemment utilisée.
    lru_precache    : cache + PRÉCACHE PRÉDICTIF. Avant chaque heure, le nœud devine les situations
                      les plus probables à cette heure-là (d'après l'historique) et les calcule
                      d'avance pendant qu'il est libre.

Ce module est une COUCHE entre le simulateur et le scheduler : il baisse le coût des tâches
qui sont des hits, et ajoute des tâches "precache" (priorité 0) pour les calculs d'avance.

Deux tâches sont mises en cache : "inference" et "anomaly" (elles ne dépendent que de l'état
courant de l'intersection). "features" dépend de l'historique des heures précédentes, "aggregation"
et "retraining" ne se répètent pas : on ne les met pas en cache.

UTILISATION
-----------
    python cache_manager.py --hours 240              # compare les 3 modes
    python cache_manager.py --hours 240 --explain 12 # montre 12 décisions (miss / hit)
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from simulator import SIM_CONFIG, TASK_TYPES, Task, TrafficSimulator

CACHEABLE = ("inference", "anomaly")
LOOKUP_COST = 0.05            # coût CPU pour relire une réponse mémorisée
PRECACHE_ID_START = 10_000_000


# --------------------------------------------------------------------------- #
# La clé : "à quelle situation ressemble cette tâche ?"
# --------------------------------------------------------------------------- #
def state_signature(payload: dict) -> tuple:
    """
    Regroupe les situations proches dans la même 'case' (échelle logarithmique :
    200 et 230 véhicules sont proches, 200 et 2000 non).
    Plus les cases sont grandes, plus on a de hits, mais plus la réponse réutilisée est approximative.
    """
    v = int(np.log2(payload["vehicle_count"] + 1) * 1.5)
    q = int(np.log2(payload["queue_length"] + 1))
    s = int(payload["average_speed"] // 15)
    return (v, q, s)


def make_key(task_type: str, intersection: str, payload: dict) -> tuple:
    return (task_type, intersection) + state_signature(payload)


# --------------------------------------------------------------------------- #
# Le cache LRU avec durée de validité
# --------------------------------------------------------------------------- #
@dataclass
class Entry:
    time: float                    # instant où le résultat est disponible
    source: str                    # "compute" (calculé à la demande) ou "precache"
    task_id: int                   # tâche qui l'a produit
    vehicles: Optional[float] = None   # état de la situation mémorisée (pour mesurer l'écart)


class ResultCache:
    def __init__(self, capacity: int = 128, ttl_s: float = 360.0):
        self.capacity, self.ttl_s = capacity, ttl_s
        self.data: OrderedDict = OrderedDict()

    def get(self, key, now: float) -> Optional[Entry]:
        e = self.data.get(key)
        if e is None:
            return None
        if now - e.time > self.ttl_s:          # trop vieux : résultat périmé
            del self.data[key]
            return None
        self.data.move_to_end(key)             # "récemment utilisé"
        return e

    def contains(self, key, now: float) -> bool:
        e = self.data.get(key)
        return e is not None and now - e.time <= self.ttl_s

    def put(self, key, entry: Entry) -> None:
        self.data[key] = entry
        self.data.move_to_end(key)
        while len(self.data) > self.capacity:  # plein : on oublie le moins récemment utilisé
            self.data.popitem(last=False)


# --------------------------------------------------------------------------- #
# Le prédicteur de popularité (pour le précache)
# --------------------------------------------------------------------------- #
class PopularityPredictor:
    """Pour chaque (intersection, heure de la journée, type de tâche), retient les situations
    les plus fréquentes déjà vues. Prédit les K plus fréquentes pour la prochaine fois."""

    def __init__(self, top_k: int = 1, min_share: float = 0.4):
        self.top_k = top_k
        self.min_share = min_share          # une situation n'est précalculée que si elle représente
        self.history: dict = defaultdict(Counter)   # au moins cette part des observations passées

    def observe(self, task_type, intersection, hour, signature) -> None:
        self.history[(task_type, intersection, hour)][signature] += 1

    def predict(self, task_type, intersection, hour) -> list[tuple]:
        h = self.history[(task_type, intersection, hour)]
        total = sum(h.values())
        if total < 3:                          # pas assez d'historique : on ne devine pas
            return []
        return [sig for sig, c in h.most_common(self.top_k) if c / total >= self.min_share]


# --------------------------------------------------------------------------- #
# La couche cache appliquée à tout le flux
# --------------------------------------------------------------------------- #
def apply_cache(batches, mode: str = "lru", capacity: int = 128, ttl_hours: float = 6.0,
                top_k: int = 1, min_share: float = 0.4):
    """Retourne (batches modifiés, statistiques, journal). Ne touche pas aux batches d'origine."""
    batches = copy.deepcopy(batches)
    hour_s = SIM_CONFIG["hour_duration_s"]
    cache = ResultCache(capacity, ttl_hours * hour_s)
    predictor = PopularityPredictor(top_k, min_share)
    stats, log = Counter(), []
    saved_cpu, pre_cpu, gaps = 0.0, 0.0, []
    pre_ids, used_pre_ids = set(), set()
    next_id = PRECACHE_ID_START

    def hour_base(tasks):
        return (min(t.arrival_time for t in tasks) // hour_s) * hour_s

    for i, (ts, tasks) in enumerate(batches):
        # 1) Les vraies tâches de cette heure, dans l'ordre d'arrivée
        for t in sorted(tasks, key=lambda t: t.arrival_time):
            if t.task_type not in CACHEABLE:
                continue
            stats["cacheable"] += 1
            key = make_key(t.task_type, t.intersection_id, t.payload)
            hit = cache.get(key, t.arrival_time) if mode != "no_cache" else None
            outcome = "MISS"
            if hit is not None:
                saved_cpu += t.cpu_cost - LOOKUP_COST
                t.cpu_cost = LOOKUP_COST
                t.payload["cache"] = "hit"
                if hit.source == "precache":
                    stats["hit_precache"] += 1
                    t.payload["precache_task"] = hit.task_id
                    used_pre_ids.add(hit.task_id)
                    outcome = "HIT (précache)"
                else:
                    stats["hit_reactive"] += 1
                    outcome = "HIT (cache)"
                    if hit.vehicles:
                        gaps.append(abs(t.payload["vehicle_count"] - hit.vehicles) / hit.vehicles)
            elif mode != "no_cache":
                cache.put(key, Entry(t.arrival_time, "compute", t.task_id, t.payload["vehicle_count"]))
            if t.task_type == "inference":
                log.append((ts, t.intersection_id, key[2:], outcome))
            predictor.observe(t.task_type, t.intersection_id, ts.hour, state_signature(t.payload))

        # 2) Précache : on prépare la PROCHAINE heure pendant la 2e moitié de celle-ci
        if mode == "lru_precache" and i + 1 < len(batches):
            next_ts, next_tasks = batches[i + 1]
            base_next = hour_base(next_tasks)
            pre_arrival = base_next - 0.5 * hour_s
            for inter in sorted({t.intersection_id for t in next_tasks if t.intersection_id}):
                for ttype in CACHEABLE:
                    for sig in predictor.predict(ttype, inter, next_ts.hour):
                        key = (ttype, inter) + sig
                        if cache.contains(key, pre_arrival):
                            continue                       # déjà en cache, rien à refaire
                        next_id += 1
                        cost = TASK_TYPES[ttype]["cpu"]
                        tasks.append(Task(next_id, "precache", tasks[0].node_id, inter, cost, 0,
                                          pre_arrival, base_next, 0.0, ts, False,
                                          {"for": ttype}))
                        cache.put(key, Entry(pre_arrival, "precache", next_id))
                        pre_ids.add(next_id)
                        pre_cpu += cost

    n = max(stats["cacheable"], 1)
    hits = stats["hit_reactive"] + stats["hit_precache"]
    total_cacheable_cpu = sum(t.cpu_cost for _, ts_ in batches for t in ts_ if t.task_type in CACHEABLE) + saved_cpu
    summary = {
        "mode": mode,
        "hit_%": 100 * hits / n,
        "hit_cache_%": 100 * stats["hit_reactive"] / n,
        "hit_precache_%": 100 * stats["hit_precache"] / n,
        "cpu_economise_%": 100 * saved_cpu / max(total_cacheable_cpu, 1e-9),
        "cpu_precache": pre_cpu,
        "precache_utile_%": 100 * len(used_pre_ids) / len(pre_ids) if pre_ids else float("nan"),
        "ecart_etat_moy_%": 100 * float(np.mean(gaps)) if gaps else float("nan"),
    }
    return batches, summary, log


MODES = ["no_cache", "lru", "lru_precache"]


# --------------------------------------------------------------------------- #
# Comparaison + effet sur le scheduling
# --------------------------------------------------------------------------- #
def compare(max_hours: int, explain_n: int = 0):
    from offloading import run_policy as run_offloading

    base_batches = list(TrafficSimulator().stream(max_hours))
    cache_rows, impact_rows, logs = [], [], {}
    for mode in MODES:
        batches, summary, log = apply_cache(base_batches, mode)
        cache_rows.append(summary)
        logs[mode] = log
        for policy in ("local_only", "offload_deadline"):
            r = run_offloading(policy, batches)
            impact_rows.append({"cache": mode, "politique": policy,
                                "latence_moy_s": r["latence_moy_s"], "retard_%": r["retard_%"],
                                "retard_alertes_%": r["retard_alertes_%"],
                                "precache_a_temps_%": r.get("precache_a_temps_%", float("nan"))})
    cache_df = pd.DataFrame(cache_rows).set_index("mode").round(1)
    impact_df = pd.DataFrame(impact_rows).set_index(["cache", "politique"]).round(2)
    return cache_df, impact_df, logs


def show_decisions(log, n: int, skip_hours: int = 72) -> None:
    inter = log[0][1]
    rows = [e for e in log if e[1] == inter][skip_hours:skip_hours + n]
    print(f"\nDécisions du cache pour {inter} (tâche 'inference'), après {skip_hours} heures d'historique :")
    for ts, _, sig, outcome in rows:
        print(f"   {ts:%d/%m %Hh}  situation={sig}  ->  {outcome}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cache et précache du nœud fog")
    ap.add_argument("--hours", type=int, default=240, help="heures du CSV à rejouer")
    ap.add_argument("--explain", type=int, default=0, help="afficher N décisions du cache")
    args = ap.parse_args()

    cache_df, impact_df, logs = compare(args.hours, args.explain)
    print(f"\n=== Efficacité du cache ({args.hours} heures simulées) ===\n")
    print(cache_df.to_string())
    print("\n=== Effet sur le scheduling (tâches de l'utilisateur seulement) ===\n")
    print(impact_df.to_string())
    if args.explain:
        show_decisions(logs["lru_precache"], args.explain)