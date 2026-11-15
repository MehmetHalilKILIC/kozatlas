"""KÖZ Atlas motoru için temel doğrulama testleri:  python -m pytest -q  veya  python tests/test_harmonizer.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harmonizer as hz  # noqa: E402


def test_haversine_known_distance():
    # 1 derece enlem farkı ≈ 2πR/360 = 111.195 km (R = 6371.0 km)
    d = hz.haversine_km(37.0, 41.0, 38.0, 41.0)
    assert abs(d - 2 * np.pi * 6371.0 / 360) < 1e-6


def _row(lat, lon, t, inst, frp, sat, conf):
    base = {"latitude": lat, "longitude": lon, "acq_date": "2024-07-01", "acq_time": t,
            "instrument": inst, "frp": frp, "satellite": sat, "confidence": conf, "daynight": "D"}
    return base


def test_clustering_and_labels():
    raw = pd.DataFrame([
        _row(37.8000, 41.0000, 1040, "VIIRS", 3.0, "N", "n"),
        _row(37.8030, 41.0010, 1040, "VIIRS", 2.0, "N", "h"),
        _row(37.8010, 41.0040, 1100, "MODIS", 6.0, "A", 85),
        _row(37.9000, 41.2000, 1040, "VIIRS", 1.0, "N", "n"),   # ~20 km uzakta: ayrı olay
        _row(37.8000, 41.0000, 1400, "VIIRS", 2.0, "N", "n"),   # aynı yer, 200 dk sonra: ayrı olay
        _row(37.5000, 40.5000, 1040, "MODIS", 9.0, "A", 10),    # düşük güven: filtrelenir
    ])
    std = hz.standardize_firms(raw)
    assert list(std["conf_class"]) .count("low") == 1
    res = hz.run_pipeline(std, hz.PipelineParams(distance_km=1.2, window_min=60, min_conf_level=1))
    assert res.n_after_confidence == 5
    ev = res.events.set_index("unified_hotspot_id")
    assert len(ev) == 3
    both = ev[ev["sensor_label"] == hz.SENSOR_BOTH]
    assert len(both) == 1 and int(both["merged_pixel_count"].iloc[0]) == 3
    # VIIRS FRP'si aynı geçişte toplanır (3 + 2 = 5 MW)
    assert abs(both["frp_viirs_mw"].iloc[0] - 5.0) < 1e-9
    assert abs(res.stats["dedup_ratio"] - (1 - 3 / 5)) < 1e-9


def test_static_detection():
    days = pd.date_range("2023-01-05", periods=12, freq="30D")
    rows = [_row(37.80, 41.10, 1000, "VIIRS", 10.0, "N", "n") | {"acq_date": d.strftime("%Y-%m-%d")} for d in days]
    rows += [_row(37.20, 40.00, 1000, "VIIRS", 4.0, "N", "n") | {"acq_date": "2023-07-0%d" % i} for i in range(1, 7)]
    res = hz.run_pipeline(hz.standardize_firms(pd.DataFrame(rows)), hz.PipelineParams())
    ev = res.events
    assert ev.loc[ev["latitude"] > 37.5, "is_static_source"].all()        # yıl boyu: statik
    assert not ev.loc[ev["latitude"] < 37.5, "is_static_source"].any()    # tek ayda 6 gün: dinamik
    assert len(res.static_sites) == 1


def test_synthetic_end_to_end():
    raw = hz.SyntheticFIRMSGenerator(seed=1, fires_per_year=60).generate(2010, 2014)
    res = hz.run_pipeline(hz.standardize_firms(raw), hz.PipelineParams())
    assert len(res.events) < len(res.points)
    assert res.calibration.n_pairs > 0
    m = hz.monthly_matrix(res.events)
    assert m.shape == (5, 12)
    assert not hz.export_table(res.events).empty


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
