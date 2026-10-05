"""Configuration du noeud fog. C'est le SEUL fichier que chaque collègue modifie."""

NODE_CONFIG = {
    "node_id": "fog_node_2",
    "zone_id": 2,
    "zone_name": "Industrial East",
    "csv_path": "fog_zone_2_Industrial_East.csv",
    # Toutes les intersections de la zone sont gardées (un noeud fog surveille toute sa zone)
    "intersections": ["INT_002", "INT_006", "INT_028", "INT_035",
                      "INT_036", "INT_064", "INT_078", "INT_085"],
}

TARGET = "accident_reported"

# Colonnes supprimées : identifiants, constantes, ou fuite de données
# (fuel/emission/flow/congestion sont des conséquences ou des copies d'autres colonnes)
DROP_COLS = [
    "record_id", "city_zone", "iot_sensor_health",
    "traffic_flow_rate",      # identique à vehicle_count
    "fuel_waste_estimate", "emission_estimate",   # conséquences de la congestion
    "congestion_score", "congestion_level",       # dérivées d'autres colonnes
    "day_of_week",            # remplacé par is_weekend + hour cyclique
]

CATEGORICAL = ["road_type", "weather_condition", "road_condition", "peak_period"]

# Mesures dont on suit l'évolution dans le temps (détection de changement brutal)
DYNAMIC = ["queue_length", "average_wait_time", "average_speed", "vehicle_count"]

# Split temporel : entraînement avant cette date, test après
TRAIN_END = "2026-03-01"