"""
scheduler.py : ordonnancement (scheduling) des tâches du nœud fog.

PROBLÈME
--------
Le simulateur produit des tâches. Le nœud dispose de plusieurs machines (workers)
de vitesses différentes. Il faut décider :
    1) DANS QUEL ORDRE exécuter les tâches
    2) SUR QUELLE MACHINE

Politiques implémentées (même interface : policy(tasks, machines)) :
    fifo     : premier arrivé, premier servi, sur la première machine libre
    min_min  : à chaque tour, on exécute la tâche dont le meilleur temps de fin est le PLUS PETIT
    max_min  : à chaque tour, on exécute la tâche dont le meilleur temps de fin est le PLUS GRAND
    edf      : heuristique "Earliest Deadline First" (deadline la plus proche d'abord)

Les métaheuristiques (génétique, PSO) et l'hyperheuristique se brancheront plus tard
dans le dictionnaire POLICIES, sans rien changer d'autre.

UTILISATION
-----------
    python scheduler.py --demo          # petit exemple à 6 tâches, pour COMPRENDRE
    python scheduler.py --hours 72      # compare toutes les politiques sur le flux du simulateur
    python scheduler.py --export results.csv
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from simulator import Task, TrafficSimulator

# Machines du nœud : {nom: capacité en unités CPU par seconde simulée}.
# Elles sont volontairement HÉTÉROGÈNES, sinon min-min / max-min n'ont aucun intérêt.
# Capacités réglées pour qu'on ait de la charge en heure de pointe (sinon toutes les politiques se valent).
DEFAULT_MACHINES = {"worker_small": 0.8, "worker_big": 1.6}


# --------------------------------------------------------------------------- #
# Une machine
# --------------------------------------------------------------------------- #
@dataclass
class Machine:
    name: str
    capacity: float                 # unités CPU / seconde
    ready_time: float = 0.0         # instant où la machine devient libre
    busy_time: float = 0.0          # temps total passé à calculer
    log: list = field(default_factory=list)   # tâches exécutées, dans l'ordre

    def exec_time(self, task: Task) -> float:
        return task.cpu_cost / self.capacity

    def completion_time(self, task: Task) -> float:
        """Quand la tâche finirait si on la mettait sur cette machine maintenant."""
        return max(self.ready_time, task.arrival_time) + self.exec_time(task)

    def run(self, task: Task) -> None:
        """Exécute (simule) la tâche sur cette machine et remplit ses champs."""
        start = max(self.ready_time, task.arrival_time)   # on ne démarre pas avant l'arrivée
        duration = self.exec_time(task)
        task.assigned_node = self.name
        task.start_time = start
        task.finish_time = start + duration
        self.ready_time = task.finish_time
        self.busy_time += duration
        self.log.append(task)


def make_machines(cfg: dict | None = None) -> list[Machine]:
    return [Machine(name, cap) for name, cap in (cfg or DEFAULT_MACHINES).items()]


def best_machine(task: Task, machines: list[Machine]) -> Machine:
    """La machine sur laquelle la tâche FINIT le plus tôt."""
    return min(machines, key=lambda m: m.completion_time(task))


# --------------------------------------------------------------------------- #
# Les politiques
# --------------------------------------------------------------------------- #
def fifo(tasks: list[Task], machines: list[Machine]) -> None:
    """Ordre d'arrivée. Chaque tâche va sur la première machine libre (sans regarder sa vitesse)."""
    for t in sorted(tasks, key=lambda t: t.arrival_time):
        min(machines, key=lambda m: m.ready_time).run(t)


def _available(pending: list[Task], machines: list[Machine]) -> list[Task]:
    """
    Tâches DÉJÀ ARRIVÉES au moment où une machine peut décider.
    Sans cela, un algorithme pourrait choisir une tâche qui n'est pas encore arrivée
    et bloquer une machine pour rien (elle attendrait cette tâche en restant inactive).
    """
    now = max(min(m.ready_time for m in machines), min(t.arrival_time for t in pending))
    return [t for t in pending if t.arrival_time <= now]


def _batch_policy(tasks: list[Task], machines: list[Machine], pick_largest: bool) -> None:
    """Cœur commun de min-min et max-min (mode batch)."""
    pending = list(tasks)
    while pending:
        # pour chaque tâche disponible : sa meilleure machine et son meilleur temps de fin
        candidates = []
        for t in _available(pending, machines):
            m = best_machine(t, machines)
            candidates.append((m.completion_time(t), t, m))
        pick = max if pick_largest else min
        _, task, machine = pick(candidates, key=lambda c: c[0])
        machine.run(task)
        pending.remove(task)


def min_min(tasks: list[Task], machines: list[Machine]) -> None:
    """Petites tâches d'abord : réduit la latence moyenne, mais les grosses tâches attendent."""
    _batch_policy(tasks, machines, pick_largest=False)


def max_min(tasks: list[Task], machines: list[Machine]) -> None:
    """Grosses tâches d'abord : évite qu'elles soient bloquées, équilibre mieux les machines."""
    _batch_policy(tasks, machines, pick_largest=True)


def edf(tasks: list[Task], machines: list[Machine]) -> None:
    """Heuristique : deadline la plus proche d'abord, sur la machine qui finit le plus tôt."""
    pending = list(tasks)
    while pending:
        task = min(_available(pending, machines), key=lambda t: t.deadline)
        best_machine(task, machines).run(task)
        pending.remove(task)


def priority_edf(tasks: list[Task], machines: list[Machine]) -> None:
    """
    Les ALERTES d'abord : priorité stricte (3 > 2 > 1 > 0), puis deadline la plus proche
    à l'intérieur d'un même niveau. Une tâche critique passe toujours devant une tâche normale,
    et le réentraînement (priorité 0) ne passe que quand il n'y a rien d'autre à faire.
    """
    pending = list(tasks)
    while pending:
        task = min(_available(pending, machines), key=lambda t: (-t.priority, t.deadline))
        best_machine(task, machines).run(task)
        pending.remove(task)


POLICIES: dict[str, Callable[[list[Task], list[Machine]], None]] = {
    "fifo": fifo,
    "min_min": min_min,
    "max_min": max_min,
    "edf": edf,
    "priority_edf": priority_edf,
    # "genetic": genetic,        # métaheuristique (à venir)
    # "hyper": hyper_heuristic,  # hyperheuristique (à venir)
}


# --------------------------------------------------------------------------- #
# Exécution d'une politique sur tout le flux + métriques
# --------------------------------------------------------------------------- #
def run_policy(name: str, batches: list[tuple[pd.Timestamp, list[Task]]],
               machine_cfg: dict | None = None) -> dict:
    """Applique la politique heure par heure (batch). L'état des machines se conserve d'une heure à l'autre."""
    batches = copy.deepcopy(batches)          # chaque politique travaille sur ses propres copies
    machines = make_machines(machine_cfg)
    policy = POLICIES[name]
    for _, tasks in batches:
        policy(tasks, machines)
    return evaluate(name, [t for _, tasks in batches for t in tasks], machines)


def evaluate(name: str, tasks: list[Task], machines: list[Machine]) -> dict:
    latency = np.array([t.latency for t in tasks])
    wait = np.array([t.start_time - t.arrival_time for t in tasks])
    missed = np.array([t.deadline_missed for t in tasks])
    urgent = np.array([t.priority >= 2 for t in tasks])
    critical = np.array([t.priority >= 3 for t in tasks])     # les vraies alertes
    horizon = max(t.finish_time for t in tasks) - min(t.arrival_time for t in tasks)
    return {
        "policy": name,
        "tasks": len(tasks),
        "latence_moy_s": latency.mean(),
        "latence_p95_s": np.percentile(latency, 95),
        "attente_moy_s": wait.mean(),
        "retard_%": 100 * missed.mean(),
        "retard_urgentes_%": 100 * missed[urgent].mean() if urgent.any() else 0.0,
        "retard_alertes_%": 100 * missed[critical].mean() if critical.any() else 0.0,
        "attente_alertes_s": wait[critical].mean() if critical.any() else 0.0,
        "makespan_s": horizon,
        "utilisation_%": 100 * np.mean([m.busy_time / horizon for m in machines]),
    }


def benchmark(max_hours: int | None = None, machine_cfg: dict | None = None) -> pd.DataFrame:
    sim = TrafficSimulator()
    batches = list(sim.stream(max_hours))     # le MÊME flux pour toutes les politiques
    rows = [run_policy(name, batches, machine_cfg) for name in POLICIES]
    return pd.DataFrame(rows).set_index("policy").round(2)


# --------------------------------------------------------------------------- #
# Démo pédagogique : 6 tâches, 2 machines
# --------------------------------------------------------------------------- #
def demo() -> None:
    costs = {"A": 2, "B": 12, "C": 3, "D": 1, "E": 8, "F": 4}     # coût CPU de chaque tâche
    ts = pd.Timestamp("2026-03-01")
    base = [Task(i + 1, n, "demo", None, c, 1, 0.0, 1e9, 1.0, ts, False)
            for i, (n, c) in enumerate(costs.items())]                # (task_type = lettre A..F)
    print("Tâches (toutes arrivent à t=0) :", costs)
    print("Machines : small = 1 unité/s, big = 2 unités/s\n")

    for name in POLICIES:
        tasks = copy.deepcopy(base)
        machines = make_machines({"small": 1.0, "big": 2.0})
        POLICIES[name](tasks, machines)
        print(f"--- {name.upper()} ---")
        for m in machines:
            seq = "  ".join(f"{t.task_type}({t.start_time:.1f}->{t.finish_time:.1f})" for t in m.log)
            print(f"  {m.name:<6}: {seq}")
        print(f"  fin de la dernière tâche (makespan) = {max(t.finish_time for t in tasks):.1f} s"
              f" | fin moyenne = {np.mean([t.finish_time for t in tasks]):.2f} s\n")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Comparaison des politiques de scheduling")
    ap.add_argument("--demo", action="store_true", help="exemple à 6 tâches")
    ap.add_argument("--hours", type=int, default=72, help="heures du CSV à rejouer")
    ap.add_argument("--export", type=str, default=None, help="exporter les résultats en CSV")
    args = ap.parse_args()

    if args.demo:
        demo()
    else:
        results = benchmark(args.hours)
        print(f"Machines : {DEFAULT_MACHINES}  |  {args.hours} heures simulées\n")
        print(results.to_string())
        if args.export:
            results.to_csv(args.export)
            print(f"\nExporté -> {args.export}")