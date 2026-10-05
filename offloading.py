"""
offloading.py : tache win mch n5dmha a,dy weli and mariem ou yessmine

PROBLÈME
--------
Quand le nœud local est saturé (rafale en heure de pointe), il peut ENVOYER une tâche :
    - à un nœud fog VOISIN (réseau rapide, mais le voisin a sa propre charge)
    - au CLOUD (très puissant, mais réseau lent et loin)
ou la garder en LOCAL (aucun réseau, mais la file d'attente est peut-être longue).

COMMENT LE NŒUD DÉCIDE
----------------------
Pour chaque tâche, on calcule l'heure de fin prévue pour CHAQUE option :

    local   : fin = max(machine libre, arrivée) + coût / vitesse
    distant : fin = envoi (latence + taille/débit)  +  attente et calcul chez le distant
                    + retour du résultat (latence)

puis la politique choisit une option.

Politiques :
    local_only        : tout reste en local (référence, = priority_edf du scheduler)
    offload_min_time  : on choisit TOUJOURS l'option qui finit le plus tôt
    offload_deadline  : on reste en local si la deadline est respectée, sinon on décharge
                        vers l'option distante qui finit le plus tôt
Dans les deux politiques avec offloading, les tâches très lourdes (réentraînement) partent au cloud.

L'ordre de traitement reste celui du scheduler : alertes d'abord, puis deadline la plus proche.

UTILISATION
-----------
    python offloading.py --hours 72                 # compare les politiques
    python offloading.py --hours 24 --explain 5     # affiche 5 décisions détaillées
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scheduler import DEFAULT_MACHINES, Machine, _available, best_machine, evaluate, make_machines
from simulator import SIM_CONFIG, Task, TrafficSimulator

HEAVY_CPU = 50.0   # une tâche au-dessus de ce coût est "lourde" -> cloud


# --------------------------------------------------------------------------- #
# Un site distant (voisin ou cloud)
# --------------------------------------------------------------------------- #
@dataclass
class Site:
    name: str
    kind: str                    # "neighbor" ou "cloud"
    machines: list[Machine]
    latency_s: float             # délai réseau, dans un sens (secondes)
    bandwidth_kb_s: float        # débit (Ko par seconde)
    background_util: float = 0.0 # part de la capacité déjà utilisée par SES propres tâches

    def __post_init__(self):
        self.bg_jobs: list[tuple[float, float]] = []   # (arrivée, coût) de ses propres tâches
        self.bg_idx = 0

    def sync(self, now: float) -> None:
        """
        Fait travailler le site sur SES propres tâches arrivées jusqu'à l'instant `now`.
        Ainsi, quand on estime un envoi à l'instant t, on voit la file du voisin telle qu'elle
        est à t, et non sa charge de toute l'heure à venir (on ne revient jamais en arrière).
        """
        while self.bg_idx < len(self.bg_jobs) and self.bg_jobs[self.bg_idx][0] <= now:
            arrival, cost = self.bg_jobs[self.bg_idx]
            m = min(self.machines, key=lambda m: m.ready_time)
            m.ready_time = max(m.ready_time, arrival) + cost / m.capacity
            self.bg_idx += 1

    def estimate(self, task: Task):
        """Heure de fin totale si on envoie la tâche ici (envoi + file + calcul + retour)."""
        send = self.latency_s + task.data_size_kb / self.bandwidth_kb_s
        remote_arrival = task.arrival_time + send
        best = None
        for m in self.machines:
            start = max(m.ready_time, remote_arrival)
            finish = start + m.exec_time(task)
            if best is None or finish < best[0]:
                best = (finish, m, start)
        finish, machine, start = best
        return finish + self.latency_s, machine, start      # + retour du résultat

    def run(self, task: Task, machine: Machine, start: float) -> None:
        duration = machine.exec_time(task)
        task.assigned_node = f"{self.name}/{machine.name}"
        task.start_time = start
        task.finish_time = start + duration + self.latency_s
        machine.ready_time = start + duration
        machine.busy_time += duration
        machine.log.append(task)


def default_sites() -> list[Site]:
    """Deux voisins déjà bien occupés + un cloud puissant mais lointain."""
    return [
        Site("fog_node_1", "neighbor", make_machines({"small": 0.8, "big": 1.6}), 0.3, 1000.0, 0.45),
        Site("fog_node_3", "neighbor", make_machines({"small": 0.8, "big": 1.6}), 0.4, 1000.0, 0.60),
        Site("cloud", "cloud", make_machines({f"vm{i}": 10.0 for i in range(4)}), 1.5, 200.0, 0.0),
    ]


def add_background(site: Site, base: float, hour_s: float, rng: np.random.Generator) -> None:
    """Simule le travail PROPRE du voisin pendant cette heure (il n'est pas vide !)."""
    if site.background_util <= 0:
        return
    total = site.background_util * sum(m.capacity for m in site.machines) * hour_s
    jobs, spent = [], 0.0
    while spent < total:
        cost = rng.uniform(2.0, 6.0)
        spent += cost
        jobs.append((base + rng.uniform(0, hour_s), cost))
    site.bg_jobs.extend(sorted(jobs))        # elles seront traitées au fil du temps (voir Site.sync)


# --------------------------------------------------------------------------- #
# La décision
# --------------------------------------------------------------------------- #
def decide(task: Task, local: list[Machine], sites: list[Site], policy: str):
    """Retourne (choix, toutes_les_options). Un choix = (nom, fin_prevue, machine, site_ou_None, start)."""
    lm = best_machine(task, local)
    options = [("local", lm.completion_time(task), lm, None, None)]
    for s in sites:
        s.sync(task.arrival_time)                    # état du voisin au moment de l'arrivée de la tâche
        finish, m, start = s.estimate(task)
        options.append((s.name, finish, m, s, start))

    if policy == "local_only" or task.task_type == "precache":   # le précache reste toujours en local
        return options[0], options

    remote = options[1:]
    if task.cpu_cost >= HEAVY_CPU:                              # tâche lourde -> cloud
        return next(o for o in remote if o[3].kind == "cloud"), options

    if policy == "offload_min_time":
        return min(options, key=lambda o: o[1]), options

    if policy == "offload_deadline":
        if options[0][1] <= task.deadline:                      # le local suffit
            return options[0], options
        ok = [o for o in remote if o[1] <= task.deadline]       # options distantes qui tiennent la deadline
        return min(ok or options, key=lambda o: o[1]), options

    raise ValueError(f"politique inconnue : {policy}")


def explain(task: Task, chosen, options) -> None:
    print(f"\nTâche #{task.task_id} {task.task_type} (priorité {task.priority}) "
          f"coût={task.cpu_cost:.1f}  deadline = +{task.deadline - task.arrival_time:.1f} s après son arrivée")
    for name, finish, *_ in options:
        rel = finish - task.arrival_time
        mark = "OK " if finish <= task.deadline else "TROP TARD"
        star = "  <-- choisi" if name == chosen[0] else ""
        print(f"   {name:<11}: finit à +{rel:5.1f} s  {mark}{star}")


# --------------------------------------------------------------------------- #
# Exécution sur tout le flux
# --------------------------------------------------------------------------- #
def run_policy(policy: str, batches, local_cfg: dict | None = None,
               seed: int = 7, explain_n: int = 0) -> dict:
    batches = copy.deepcopy(batches)
    local = make_machines(local_cfg or DEFAULT_MACHINES)
    sites = default_sites()
    rng = np.random.default_rng(seed)           # même charge de fond pour toutes les politiques
    hour_s = SIM_CONFIG["hour_duration_s"]
    shown = 0

    for _, tasks in batches:
        base = (min(t.arrival_time for t in tasks) // hour_s) * hour_s
        for s in sites:
            add_background(s, base, hour_s, rng)

        pending = list(tasks)
        while pending:
            task = min(_available(pending, local), key=lambda t: (-t.priority, t.deadline))
            chosen, options = decide(task, local, sites, policy)
            name, _, machine, site, start = chosen
            if site is None:
                machine.run(task)
            else:
                site.run(task, machine, start)
            if shown < explain_n and task.priority >= 3 and policy == "offload_deadline":
                explain(task, chosen, options)
                shown += 1
            pending.remove(task)

    all_tasks = [t for _, tasks in batches for t in tasks]
    # Les tâches "precache" occupent les machines mais ne comptent pas dans les mesures de l'utilisateur
    user_tasks = [t for t in all_tasks if t.task_type != "precache"]
    row = evaluate(policy, user_tasks, local)
    n = len(user_tasks)
    row["voisins_%"] = 100 * sum(t.assigned_node.startswith("fog_node") for t in user_tasks) / n
    row["cloud_%"] = 100 * sum(t.assigned_node.startswith("cloud") for t in user_tasks) / n

    # Un "hit précache" n'est valable que si le calcul d'avance était TERMINÉ à l'arrivée de la tâche
    by_id = {t.task_id: t for t in all_tasks}
    ready = [by_id[t.payload["precache_task"]].finish_time <= t.arrival_time
             for t in user_tasks if t.payload.get("precache_task")]
    if ready:
        row["precache_a_temps_%"] = 100 * float(np.mean(ready))
    return row


POLICY_NAMES = ["local_only", "offload_min_time", "offload_deadline"]


def benchmark(max_hours: int | None = None, local_cfg: dict | None = None, explain_n: int = 0) -> pd.DataFrame:
    batches = list(TrafficSimulator().stream(max_hours))     # le MÊME flux pour tous
    rows = [run_policy(p, batches, local_cfg, explain_n=explain_n) for p in POLICY_NAMES]
    cols = ["latence_moy_s", "retard_%", "retard_alertes_%", "attente_alertes_s",
            "voisins_%", "cloud_%"]
    return pd.DataFrame(rows).set_index("policy")[cols].round(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Comparaison des politiques d'offloading")
    ap.add_argument("--hours", type=int, default=72, help="heures du CSV à rejouer")
    ap.add_argument("--explain", type=int, default=0, help="afficher N décisions détaillées (alertes)")
    ap.add_argument("--export", type=str, default=None)
    args = ap.parse_args()

    results = benchmark(args.hours, explain_n=args.explain)
    print(f"\nNœud local : {DEFAULT_MACHINES} | {args.hours} heures simulées\n")
    print(results.to_string())
    if args.export:
        results.to_csv(args.export)
        print(f"\nExporté -> {args.export}")