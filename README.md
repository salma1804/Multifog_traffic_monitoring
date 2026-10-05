# MultiFog Traffic Monitoring

Trois nœuds fog (un par zone urbaine) analysent localement le trafic, détectent les accidents et
se confirment mutuellement les alertes. **Le code est identique pour les trois nœuds** : chacun
ne change que sa configuration (`config.py`) et ses propres données.

## Comment ça marche (en 30 secondes)

Chaque heure, un nœud traite les mesures de ses intersections :

```
mesures de l'heure -> anomaly.py (le gardien) -> porte -> model.py (le médecin) -> étiquette
```

| Étiquette | Signification | Action |
|---|---|---|
| `critical_accident` | trafic anormal ET le modèle voit un accident | alerte immédiate aux voisins, qui votent |
| `congestion` | trafic anormal mais pas d'accident | zone en « surveillance » |
| `normal` | rien d'anormal | rien |

Les nœuds s'envoient aussi un résumé de leur zone, d'où une décision régionale
(`normal` / `watch` / `global_alert`). Un voisin en panne est détecté et signalé, jamais ignoré.

## Les fichiers

| Fichier | Rôle |
|---|---|
| `config.py` | **réglages du nœud, le seul fichier que chacun modifie** |
| `preprocessing.py` | charge le CSV et fabrique les features (identique pour tous) |
| `model.py` | entraîne et charge le modèle de risque d'accident |
| `anomaly.py` | détecteur d'anomalies (le gardien) |
| `node_engine.py` | le cerveau du nœud : enchaîne gardien, modèle, étiquette, résumé |
| `collaboration.py` | **le contrat** : messages échangés, vote de confirmation, décision régionale |
| `api.py` | le serveur (FastAPI) : ce qui fait de ce code un vrai nœud |
| `peer_stub.py` | faux voisins pour tester sans attendre les autres |
| `demo_cluster.py` | lance 3 nœuds en local et montre alerte, vote et panne |
| `test_collaboration.py` | tests automatiques |
| `simulator.py`, `scheduler.py`, `offloading.py`, `cache_manager.py` | simulation du scheduling, de l'offloading et du cache (indépendants du serveur) |

## 1. Installation

```
git clone <url-du-depot>
cd <dossier>
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac : source venv/bin/activate)
pip install -r requirements.txt
```

## 2. CE QUE TU DOIS CHANGER (et rien d'autre)

**a) Ton fichier de données.** Place ton CSV de zone à côté du code. Il n'est pas sur Git
(voir `.gitignore`) : chacun garde le sien.

**b) `config.py`, uniquement le bloc `NODE_CONFIG` :**

```python
NODE_CONFIG = {
    "node_id": "fog_node_1",                 # EXACTEMENT fog_node_1, fog_node_2 ou fog_node_3
    "zone_id": 1,
    "zone_name": "Nom de ta zone",
    "csv_path": "fog_zone_1_xxx.csv",        # chemin RELATIF, CSV dans le même dossier
    "intersections": ["INT_xxx", "INT_xxx"], # les intersections de ta zone
}
```

**c) Les voisins**, par variable d'environnement (pas dans le code) :

```
# Windows PowerShell
$env:FOG_PEERS="fog_node_1=http://192.168.1.10:8001,fog_node_3=http://192.168.1.12:8003"

# Linux / Mac
export FOG_PEERS="fog_node_1=http://192.168.1.10:8001,fog_node_3=http://192.168.1.12:8003"
```

Mets les adresses IP des machines de tes amis (`ipconfig` sous Windows). Ton propre nom est
ignoré s'il figure dans la liste : tu peux donc donner la même liste aux trois.

### Ce qu'il ne faut PAS modifier sans l'accord des trois

- `TARGET`, `DROP_COLS`, `CATEGORICAL`, `DYNAMIC`, `TRAIN_END` dans `config.py`
- `preprocessing.py`
- `collaboration.py` (le contrat des messages ; si on le change, incrémenter `SCHEMA_VERSION`)

Si ces éléments diffèrent d'un nœud à l'autre, les colonnes ne correspondent plus et les voisins
répondent « unable » au lieu de confirmer ou rejeter une alerte.

## 3. Entraîner TON modèle

Chaque nœud entraîne ses modèles sur ses propres données (étapes à refaire après tout changement
de `config.py`, de `preprocessing.py` ou des modèles) :

```
python model.py        # crée model_<ton_node_id>.joblib
python anomaly.py      # crée anomaly_<ton_node_id>.joblib
```

## 4. Vérifier que tout marche

```
pytest -q                  # les tests (ils ignorent ce qui nécessite les .joblib s'ils manquent)
python node_engine.py      # compare la cascade au modèle seul sur la période de test
python demo_cluster.py     # démo à 3 nœuds en local (2 faux voisins), nécessite les 2 .joblib
```

## 5. Lancer ton nœud

```
uvicorn api:app --host 0.0.0.0 --port 8001       # port : 8001, 8002 ou 8003 selon ton nœud
```

- `0.0.0.0` : indispensable pour que les autres machines puissent te joindre (`127.0.0.1` n'écoute que ta machine).
- Pare-feu Windows : autoriser le port choisi en entrée.
- Vérification : ouvre `http://TON_IP:8001/health` dans un navigateur, tu dois voir `"status": "ok"`.
- Documentation interactive de l'API : `http://TON_IP:8001/docs`.

Se placer sur une heure puis la traiter (tout le monde doit utiliser la même heure) :

```
POST /seek   {"to": "2026-03-02T05:00:00"}
POST /tick                                  # traite l'heure suivante, envoie alerte et résumé
GET  /decision                              # décision régionale
GET  /events                                # alertes émises et reçues
GET  /peers                                 # résumés reçus des voisins
```

Durée de validité d'un résumé reçu : variable `FOG_TTL_S` (120 secondes par défaut).

**Attention : l'API n'a aucune authentification.** Utilise un réseau privé (même Wi-Fi) ou un VPN,
jamais Internet.

## 6. Travailler ensemble avec Git

```
git checkout -b ma-fonctionnalite
git add <fichiers>
git commit -m "message clair"
git push origin ma-fonctionnalite       # puis ouvrir une Pull Request, relue par un autre avant de fusionner
```

Règles :
- `main` doit toujours passer `pytest -q`.
- Ne jamais pousser de `.csv`, de `.joblib` ni de journaux (le `.gitignore` s'en charge).
- Ne pas modifier `config.py` d'un ami : chacun garde son bloc `NODE_CONFIG`. En cas de conflit sur ce fichier, garder **sa** version du bloc.

## Problèmes fréquents

| Symptôme | Cause probable |
|---|---|
| `FileNotFoundError ... .joblib` | lancer `python model.py` puis `python anomaly.py` |
| `relance python anomaly.py` | ancien fichier `.joblib` : relancer `python anomaly.py` |
| Voisin « injoignable » | mauvaise IP, pare-feu, voisin lancé avec `127.0.0.1`, ou `FOG_PEERS` mal écrit |
| Verdict `unable` | `preprocessing.py` ou `config.py` différent entre les nœuds, ou alerte sans mesures (faux voisin) |
| `nœud inconnu` (erreur 422) | le `node_id` de l'expéditeur n'est pas dans `FOG_PEERS` |
| Port déjà utilisé | un ancien nœud tourne encore : le fermer ou changer de port |
| `s'est arrêté` dans la démo | lire `logs_<nom>.txt` |

## Pas encore fait

Authentification de l'API, branchement du scheduler / de l'offloading / du cache dans le serveur,
Docker, intégration continue (GitHub Actions), service provisioning.