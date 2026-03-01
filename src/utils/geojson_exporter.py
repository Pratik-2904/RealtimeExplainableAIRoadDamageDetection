"""
GeoJSON Exporter — Auditor Compliance Report
============================================
Exports the TSCM asset inventory as a standards-compliant GeoJSON FeatureCollection.

Each pothole becomes a single GeoJSON Point feature with:
  • Geometry: [Longitude, Latitude] (WGS-84 — GIS standard)
  • Properties: IRC:82-2015 severity, metric area (cm²), depth (mm),
                class, detection count, unique ID.

Edge-to-Cloud hybrid:
  • This report is designed to be uploaded to a GIS dashboard (Cloud)
    while the safety branch operates locally on the Jetson Orin Nano.
"""

import json
import os
from datetime import datetime


def assets_to_geojson(inventory: list[dict], output_path: str = "road_assets.geojson") -> str:
    """
    Convert a TSCM inventory list to a GeoJSON FeatureCollection.

    Parameters
    ----------
    inventory : list[dict]
        Output of CDKFTracker.get_inventory().
    output_path : str
        Where to write the .geojson file.

    Returns
    -------
    str  path to the written file.
    """
    features = []

    for asset in inventory:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                # GeoJSON uses [lon, lat] order (WGS-84)
                "coordinates": [asset["longitude"], asset["latitude"]],
            },
            "properties": {
                "id": asset["id"],
                "class": asset.get("class", "Unknown"),
                "severity_irc82": asset.get("severity_irc82", "Unknown"),
                "maintenance_action": asset.get("maintenance_action", "Unknown"),
                "area_cm2": round(asset.get("area_cm2", 0.0), 4),
                "depth_mm": round(asset.get("depth_mm", 0.0), 2),
                "frames_tracked": asset.get("frames_tracked", 1),
                "total_detections": asset.get("total_detections", 1),
                "generated_at": datetime.utcnow().isoformat() + "Z",
            },
        }
        features.append(feature)

    geojson_doc = {
        "type": "FeatureCollection",
        "name": "RoadDamageInventory_IRC82-2015",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson_doc, f, indent=2)

    print(f"[GeoJSON] Exported {len(features)} asset(s) → {output_path}")
    return output_path


# ─── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.utils.tscm import CDKFTracker
    import numpy as np

    tracker = CDKFTracker()
    tracker.update([{
        "gps": (37.7749, -122.4194),
        "area": 120.5,
        "depth": 55.0,
        "confidence": 0.93,
        "distance": 4.0,
        "class_name": "D40_Pothole",
    }])
    tracker.update([{
        "gps": (37.77491, -122.41941),
        "area": 130.0,
        "depth": 60.0,
        "confidence": 0.97,
        "distance": 2.0,
        "class_name": "D40_Pothole",
    }])
    tracker.update([{
        "gps": (37.7760, -122.4200),  # Different pothole
        "area": 40.0,
        "depth": 20.0,
        "confidence": 0.75,
        "distance": 8.0,
        "class_name": "D00_Longitudinal_Crack",
    }])

    inventory = tracker.get_inventory()
    path = assets_to_geojson(inventory, output_path="/tmp/road_assets.geojson")

    with open(path) as f:
        doc = json.load(f)
    print(f"✓  Features in GeoJSON: {len(doc['features'])}")
    for feat in doc["features"]:
        p = feat["properties"]
        print(f"   [{p['id']}] {p['class']}  severity={p['severity_irc82']}  depth={p['depth_mm']} mm")
