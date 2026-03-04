"""
TSCM  —  Temporal-Spatial Consistency Module
=============================================
Novelty USP: converts repeated per-frame detections into deduplicated "Master Assets".

Algorithm
---------
For each incoming detection we either:
  • Match it to an existing asset (Haversine distance < threshold).
  • Start a new asset track.

State update uses a Confidence-and-Distance-weighted Kalman Filter (CDKF):
  • High-confidence, close-range frames dominate the running average.
  • Noisy far-range frames contribute little weight.

Tracking window: 20–40 frames is the design target for highway speeds.
Output: a single GeoJSON-ready inventory of unique assets.
"""

import json
import numpy as np
from filterpy.kalman import KalmanFilter


# ─── IRC:82-2015 severity table ───────────────────────────────────────────────
def irc_severity(depth_mm: float) -> tuple[str, str]:
    if depth_mm < 25:
        return "Low", "Routine Monitoring"
    elif depth_mm <= 50:
        return "Medium", "Patching / Sealing"
    else:
        return "High", "Emergency Repair"


# ─── Individual asset track ────────────────────────────────────────────────────
class AssetTrack:
    """
    One unique road-defect asset, averaging measurements across 20-40 frames.
    Uses a 4-state Kalman Filter: [lat, lon, area_cm2, depth_mm].
    """

    def __init__(self, asset_id: int, det: dict) -> None:
        self.asset_id = asset_id
        self.class_name = det.get("class_name", "D40_Pothole")
        self.hits = 1
        self.frames_tracked = 1
        self.last_distance = det.get("distance", 0.0)

        # Kalman filter state: [lat, lon, area, depth]
        self.kf = KalmanFilter(dim_x=4, dim_z=4)
        self.kf.F = np.eye(4)          # state transition (static world position)
        self.kf.H = np.eye(4)          # measurement function
        self.kf.R = np.eye(4) * 1e-4   # measurement noise (GPS ≈ cm-level)
        self.kf.Q = np.eye(4) * 1e-6   # process noise (pothole doesn't move)
        self.kf.P = np.eye(4) * 0.1    # initial uncertainty
        self.kf.x = np.array([
            det["gps"][0],
            det["gps"][1],
            det.get("area", 0.0),
            det.get("depth", 0.0),
        ], dtype=np.float64).reshape(4, 1)

        # Tracking weights for running CDKF average
        self._total_weight = self._weight(det)

    @staticmethod
    def _weight(det: dict) -> float:
        """confidence / distance  →  closer, surer frames get more weight."""
        conf = det.get("confidence", 0.5)
        dist = det.get("distance", 10.0)
        return conf / (dist + 1e-5)

    def update(self, det: dict) -> None:
        w = self._weight(det)
        # Scale measurement noise inversely with weight (trusted frames → low noise)
        scale = 1.0 / (w + 1e-5)
        self.kf.R = np.eye(4) * scale * 1e-4

        z = np.array([
            det["gps"][0],
            det["gps"][1],
            det.get("area", 0.0),
            det.get("depth", 0.0),
        ], dtype=np.float64).reshape(4, 1)

        self.kf.predict()
        self.kf.update(z)

        self._total_weight += w
        self.hits += 1
        self.frames_tracked += 1
        self.last_distance = det.get("distance", self.last_distance)

    @property
    def state(self) -> np.ndarray:
        return self.kf.x.flatten()

    def to_dict(self) -> dict:
        lat, lon, area, depth = self.state
        sev, action = irc_severity(float(depth))
        return {
            "id": self.asset_id,
            "class": self.class_name,
            "latitude": float(lat),
            "longitude": float(lon),
            "area_cm2": float(area),
            "depth_mm": float(depth),
            "severity_irc82": sev,
            "maintenance_action": action,
            "frames_tracked": self.frames_tracked,
            "total_detections": self.hits,
            "last_distance": getattr(self, "last_distance", 0.0)
        }


# ─── Multi-asset tracker ───────────────────────────────────────────────────────
class CDKFTracker:
    """
    Manages all AssetTracks.  One pothole → one map pin.

    Parameters
    ----------
    match_threshold_meters : float
        Max Haversine distance to consider two GPS coords as the same asset.
    max_frames_unmatched : int
        Prune tracks that haven't been matched for this many frames.
    """

    def __init__(
        self,
        match_threshold_meters: float = 3.0,
        max_frames_unmatched: int = 40,
    ) -> None:
        self.assets: list[AssetTrack] = []
        self.archived_assets: list[AssetTrack] = []  # save long-term assets that exit the view
        self.next_id = 0
        self.match_threshold_meters = match_threshold_meters
        self.max_frames_unmatched = max_frames_unmatched
        self._unmatched_counters: dict[int, int] = {}

    @staticmethod
    def _haversine(c1: tuple, c2: tuple) -> float:
        """Returns distance in metres between two (lat, lon) pairs."""
        lat1, lon1 = np.radians(c1)
        lat2, lon2 = np.radians(c2)
        a = (np.sin((lat2 - lat1) / 2) ** 2
             + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
        return 6_378_137.0 * 2 * np.arcsin(np.sqrt(a))

    def update(self, detections: list[dict]) -> None:
        """
        Feed one frame of detections.  Each det must have:
          gps=(lat, lon), area, depth, confidence, distance, class_name (optional).
        """
        matched_ids = set()

        for det in detections:
            gps = det["gps"]
            best_idx, best_dist = -1, float("inf")

            for i, track in enumerate(self.assets):
                d = self._haversine(gps, (track.state[0], track.state[1]))
                if d < best_dist and d < self.match_threshold_meters:
                    best_idx, best_dist = i, d

            if best_idx != -1:
                self.assets[best_idx].update(det)
                matched_ids.add(self.assets[best_idx].asset_id)
                self._unmatched_counters[self.assets[best_idx].asset_id] = 0
            else:
                new_track = AssetTrack(self.next_id, det)
                self.assets.append(new_track)
                self._unmatched_counters[self.next_id] = 0
                self.next_id += 1

        # Increment unmatched counters and prune stale tracks
        active_assets = []
        for track in self.assets:
            if track.asset_id not in matched_ids:
                self._unmatched_counters[track.asset_id] = (
                    self._unmatched_counters.get(track.asset_id, 0) + 1
                )
            
            # If the track hasn't been seen in max_frames, archive it if it's valid
            if self._unmatched_counters.get(track.asset_id, 0) > self.max_frames_unmatched:
                if track.hits >= 1:  # Keep all detections in the archive
                    self.archived_assets.append(track)
            else:
                active_assets.append(track)
                
        self.assets = active_assets

    def get_inventory(self) -> list[dict]:
        """Returns the final deduplicated list of unique road-defect assets."""
        all_assets = self.assets + self.archived_assets
        return [t.to_dict() for t in all_assets]


# ─── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tracker = CDKFTracker(match_threshold_meters=3.0, max_frames_unmatched=40)

    # Simulate 30 frames as a vehicle approaches the same pothole
    base_lat, base_lon = 37.77000, -122.41000
    for frame in range(30):
        dist = 15.0 - frame * 0.4          # vehicle closes from 15 m → 3 m
        conf = min(0.5 + frame * 0.02, 0.99)  # confidence rises as vehicle approaches
        gps_noise = np.random.normal(0, 5e-6, 2)  # ≈ sub-metre GPS noise
        tracker.update([{
            "gps": (base_lat + gps_noise[0], base_lon + gps_noise[1]),
            "area": 80.0 + frame * 1.0,
            "depth": 35.0 + frame * 0.5,
            "confidence": conf,
            "distance": max(dist, 1.0),
            "class_name": "D40_Pothole",
        }])

    inv = tracker.get_inventory()
    assert len(inv) == 1, f"Expected 1 asset, got {len(inv)}"
    a = inv[0]
    print(f"✓  Unique assets : {len(inv)}")
    print(f"   Frames tracked: {a['frames_tracked']}")
    print(f"   Total hits     : {a['total_detections']}")
    print(f"   Depth avg (mm) : {a['depth_mm']:.1f}")
    print(f"   Severity       : {a['severity_irc82']} → {a['maintenance_action']}")
    print(f"   GPS            : ({a['latitude']:.6f}, {a['longitude']:.6f})")
