"""
simulator.py : ijib les donnes ml .csv khaw

IDÉE
----
Le CSV est un historique. Un nœud fog, lui, reçoit un flux en direct.
Ce module REJOUE le CSV heure par heure et transforme chaque mesure en
TÂCHES (features, inférence, anomalie, agrégation, réentraînement) avec :
  - un coût CPU          (pour le scheduling : min-min, max-min, ...)
  - une priorité         (0 = basse ... 3 = critique)
  - un instant d'arrivée (en secondes simulées, avec rafales en heure de pointe)
  - une deadline         (pour mesurer les tâches en retard)
  - une taille de données (pour calculer le coût de transfert lors de l'offloading)

Exemple : une mesure arrive à 07:00 pour INT_006

vehicle_count = 80, queue_length = 30, average_speed = 20. Voici ce que le nœud doit faire :

1. Features (le plus léger)
Transformer la mesure brute en colonnes utilisables par le modèle : variation par rapport à l'heure précédente, moyenne des 3 dernières heures, ratio, heure cyclique. C'est exactement ton preprocessing.py, appelé pour une nouvelle ligne.

2. Inférence (prédiction du risque)
Donner les features au modèle (RandomForest ou XGBoost) pour obtenir la probabilité d'accident sur cette intersection, par exemple « 0,82 ». C'est le résultat principal du nœud.

3. Anomalie
Vérifier si la situation est anormale : la file d'attente a-t-elle explosé d'un coup ? (queue_length_ratio très élevé, ou IsolationForest). La réponse est « normal » ou « alerte ».

4. Agrégation (une fois par heure pour toute la zone)
Rassembler les résultats des 8 intersections en un seul résumé : « zone Industrial East : 2 intersections en alerte, risque moyen 0,45 ». C'est ce résumé que le nœud envoie à ses collègues (les autres nœuds fog) pour la décision collaborative.

5. Réentraînement (rare et très lourd)
Réentraîner le modèle avec les nouvelles données, par exemple une fois par jour. Cela prend beaucoup de calcul mais n'est pas urgent.
UTILISATION
-----------
    from simulator import TrafficSimulator

    sim = TrafficSimulator()
    for timestamp, tasks in sim.stream():      # tâches heure par heure
        ...                                    # -> scheduler / cache / offloading

    python simulator.py --hours 24             # aperçu + statistiques
    python simulator.py --export tasks.csv     # exporte toutes les tâches
    python simulator.py --realtime --speed 20  # rejoue en (quasi) temps réel
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, asdict
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from config import NODE_CONFIG, TRAIN_END

# --------------------------------------------------------------------------- #
# Paramètres (à déplacer dans config.py si tu veux les régler par nœud)
# --------------------------------------------------------------------------- #
SIM_CONFIG = {
    "seed": 42,                  # reproductibilité : mêmes tâches à chaque lancement
    "hour_duration_s": 60.0,     # 1 heure du CSV = 60 secondes simulées
    "rush_window_ratio": 0.25,   # en heure de pointe, les tâches arrivent dans 25 % de l'heure (rafale)
    "cost_jitter": 0.20,         # variation aléatoire de ±20 % du coût CPU
    "retrain_every_hours": 24,   # une tâche de réentraînement toutes les 24 h simulées
    "use_test_period": True,     # ne rejouer que la période de test (après TRAIN_END)
    "congestion_quantile": 0.90, # file d'attente au-dessus de ce quantile = situation anormale
}

# Catalogue des types de tâches.
#   cpu       : coût de base (millions d'instructions, unité abstraite)
#   deadline  : délai accordé (secondes simulées) avant d'être considérée en retard
#   priority  : priorité de base
#   size_kb   : taille des données à transférer si la tâche est déchargée
TASK_TYPES = {
    "features":    {"cpu": 2.0,   "deadline": 20.0,   "priority": 1, "size_kb": 2.0},
    "inference":   {"cpu": 5.0,   "deadline": 15.0,   "priority": 2, "size_kb": 4.0},
    "anomaly":     {"cpu": 3.0,   "deadline": 10.0,   "priority": 2, "size_kb": 4.0},
    "aggregation": {"cpu": 4.0,   "deadline": 30.0,   "priority": 1, "size_kb": 10.0},
    "retraining":  {"cpu": 120.0, "deadline": 3600.0, "priority": 0, "size_kb": 5000.0},
}

PAYLOAD_COLS = ["vehicle_count", "queue_length", "average_speed", "average_wait_time"]


# --------------------------------------------------------------------------- #
# Une tâche
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    task_id: int
    task_type: str
    node_id: str
    intersection_id: Optional[str]
    cpu_cost: float
    priority: int                 # 0 basse, 1 normale, 2 haute, 3 critique
    arrival_time: float           # secondes simulées
    deadline: float               # instant limite (secondes simulées)
    data_size_kb: float
    timestamp: pd.Timestamp       # heure dans le dataset
    rush_hour: bool
    payload: dict = field(default_factory=dict)

    # Remplis par le scheduler / l'exécuteur
    assigned_node: Optional[str] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    @property
    def latency(self) -> Optional[float]:
        """Temps total passé dans le système (attente + exécution)."""
        return None if self.finish_time is None else self.finish_time - self.arrival_time

    @property
    def deadline_missed(self) -> Optional[bool]:
        return None if self.finish_time is None else self.finish_time > self.deadline


# --------------------------------------------------------------------------- #
# Le simulateur
# --------------------------------------------------------------------------- #
class TrafficSimulator:
    def __init__(self, node_cfg: dict = NODE_CONFIG, sim_cfg: Optional[dict] = None,
                 df: Optional[pd.DataFrame] = None):
        self.node_cfg = node_cfg
        self.cfg = {**SIM_CONFIG, **(sim_cfg or {})}
        self.rng = np.random.default_rng(self.cfg["seed"])
        self._next_id = 0

        self.df = self._prepare(df if df is not None else self._load())
        self.t0 = self.df["timestamp"].min()
        self.max_vehicles = max(float(self.df["vehicle_count"].max()), 1.0)
        # seuil de file d'attente "anormale", calculé sur toute la zone
        self.queue_threshold = float(self.df["queue_length"].quantile(self.cfg["congestion_quantile"]))

    # ---------------------------- chargement ------------------------------- #
    def _load(self) -> pd.DataFrame:
        return pd.read_csv(self.node_cfg["csv_path"], parse_dates=["timestamp"])

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["intersection_id"].isin(self.node_cfg["intersections"])].copy()
        df = df.sort_values(["timestamp", "intersection_id"]).reset_index(drop=True)
        if self.cfg["use_test_period"]:
            # le modèle est entraîné AVANT TRAIN_END : on simule la vie réelle APRÈS
            df = df[df["timestamp"] >= pd.Timestamp(TRAIN_END)].reset_index(drop=True)
        if df.empty:
            raise ValueError("Aucune donnée à simuler (vérifie intersections / TRAIN_END / csv_path).")
        return df

    # ----------------------------- création -------------------------------- #
    def _new_task(self, task_type: str, intersection: Optional[str], arrival: float,
                  timestamp: pd.Timestamp, rush: bool, priority: int,
                  load: float, payload: Optional[dict] = None) -> Task:
        spec = TASK_TYPES[task_type]
        jitter = self.rng.uniform(1 - self.cfg["cost_jitter"], 1 + self.cfg["cost_jitter"])
        cpu = spec["cpu"] * (0.8 + 0.4 * load) * jitter          # plus de trafic = un peu plus de calcul
        deadline_delay = spec["deadline"] * (0.5 if priority >= 3 else 1.0)  # critique = délai réduit
        self._next_id += 1
        return Task(
            task_id=self._next_id, task_type=task_type, node_id=self.node_cfg["node_id"],
            intersection_id=intersection, cpu_cost=round(cpu, 3), priority=priority,
            arrival_time=arrival, deadline=arrival + deadline_delay,
            data_size_kb=spec["size_kb"], timestamp=timestamp, rush_hour=rush,
            payload=payload or {},
        )

    def _arrivals(self, n: int, base: float, rush: bool) -> np.ndarray:
        """Instants d'arrivée dans l'heure. En heure de pointe : rafale (fenêtre réduite)."""
        window = self.cfg["hour_duration_s"] * (self.cfg["rush_window_ratio"] if rush else 1.0)
        return np.sort(base + self.rng.uniform(0, window, size=n))

    def _row_priority(self, row, base: int) -> int:
        """
        Priorité d'une tâche liée à une mesure.
        On utilise des signaux OBSERVABLES (véhicule d'urgence, file anormale).
        On n'utilise PAS accident_reported : c'est la cible à prédire (fuite de données).
        """
        p = base
        if getattr(row, "emergency_vehicle_detected", 0) == 1:
            p += 1
        if row.queue_length >= self.queue_threshold:
            p += 1
        return min(p, 3)

    # ------------------------------ flux ----------------------------------- #
    def stream(self, max_hours: Optional[int] = None) -> Iterator[tuple[pd.Timestamp, list[Task]]]:
        """Génère (timestamp, tâches de cette heure), triées par instant d'arrivée."""
        hour_s = self.cfg["hour_duration_s"]
        for count, (ts, group) in enumerate(self.df.groupby("timestamp", sort=True)):
            if max_hours is not None and count >= max_hours:
                return
            hours_since_start = (ts - self.t0).total_seconds() / 3600.0
            base = hours_since_start * hour_s
            rows = list(group.itertuples(index=False))
            rush = bool(group["rush_hour"].max()) if "rush_hour" in group else False

            tasks: list[Task] = []
            arrivals = self._arrivals(len(rows) * 3, base, rush)
            rng_idx = self.rng.permutation(len(arrivals))      # mélange l'ordre des types
            k = 0
            for row in rows:
                load = min(row.vehicle_count / self.max_vehicles, 1.0)
                payload = {c: float(getattr(row, c)) for c in PAYLOAD_COLS}
                for ttype in ("features", "inference", "anomaly"):
                    base_p = TASK_TYPES[ttype]["priority"]
                    tasks.append(self._new_task(
                        ttype, row.intersection_id, float(arrivals[rng_idx[k]]), ts, rush,
                        self._row_priority(row, base_p), load, payload))
                    k += 1

            # une agrégation de zone par heure, juste après les mesures
            agg_arrival = base + (self.cfg["rush_window_ratio"] if rush else 1.0) * hour_s
            tasks.append(self._new_task("aggregation", None, agg_arrival, ts, rush,
                                        TASK_TYPES["aggregation"]["priority"], 0.5,
                                        {"n_intersections": len(rows)}))

            # réentraînement périodique : tâche lourde et peu urgente
            every = self.cfg["retrain_every_hours"]
            if every and count > 0 and count % every == 0:
                tasks.append(self._new_task("retraining", None, base, ts, rush, 0, 1.0))

            tasks.sort(key=lambda t: t.arrival_time)
            yield ts, tasks

    def all_tasks(self, max_hours: Optional[int] = None) -> list[Task]:
        return [t for _, tasks in self.stream(max_hours) for t in tasks]

    # ------------------------------ temps réel ----------------------------- #
    def replay_realtime(self, speed: float = 1.0, max_hours: Optional[int] = None) -> Iterator[Task]:
        """Émet les tâches une par une en respectant les délais (speed=20 -> 20x plus vite)."""
        start_wall = time.monotonic()
        for _, tasks in self.stream(max_hours):
            for t in tasks:
                wait = t.arrival_time / speed - (time.monotonic() - start_wall)
                if wait > 0:
                    time.sleep(wait)
                yield t

    # ------------------------------ analyse -------------------------------- #
    @staticmethod
    def to_dataframe(tasks: list[Task]) -> pd.DataFrame:
        rows = []
        for t in tasks:
            d = asdict(t)
            d.pop("payload")
            rows.append(d)
        return pd.DataFrame(rows)

    def summary(self, tasks: list[Task]) -> str:
        df = self.to_dataframe(tasks)
        lines = [
            f"Noeud {self.node_cfg['node_id']} | zone {self.node_cfg['zone_name']}",
            f"Tâches générées : {len(df)} sur {df['timestamp'].nunique()} heures simulées",
            f"Seuil de file anormale (q{int(self.cfg['congestion_quantile']*100)}) : {self.queue_threshold:.1f}",
            "\nPar type :",
            df.groupby("task_type").agg(n=("task_id", "count"), cpu_moyen=("cpu_cost", "mean")).round(2).to_string(),
            "\nPar priorité :",
            df["priority"].value_counts().sort_index().to_string(),
            "\nTâches par heure : normale vs heure de pointe",
            df.groupby(["timestamp", "rush_hour"]).size().groupby("rush_hour").mean().round(1).to_string(),
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Ligne de commande
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simulateur de flux de tâches du nœud fog")
    ap.add_argument("--hours", type=int, default=None, help="nombre d'heures du CSV à rejouer")
    ap.add_argument("--export", type=str, default=None, help="chemin CSV pour exporter les tâches")
    ap.add_argument("--realtime", action="store_true", help="rejouer en temps (quasi) réel")
    ap.add_argument("--speed", type=float, default=20.0, help="accélération du mode temps réel")
    args = ap.parse_args()

    sim = TrafficSimulator()

    if args.realtime:
        for task in sim.replay_realtime(speed=args.speed, max_hours=args.hours):
            print(f"[t={task.arrival_time:8.1f}s] {task.task_type:<11} {str(task.intersection_id):<8} "
                  f"prio={task.priority} cpu={task.cpu_cost:6.2f} deadline=+{task.deadline - task.arrival_time:.0f}s")
    else:
        tasks = sim.all_tasks(max_hours=args.hours)
        print(sim.summary(tasks))
        print("\nExemple (3 premières tâches) :")
        for t in tasks[:3]:
            print(" ", t)
        if args.export:
            sim.to_dataframe(tasks).to_csv(args.export, index=False)
            print(f"\nExporté -> {args.export}")