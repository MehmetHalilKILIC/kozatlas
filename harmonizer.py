"""
KÖZ Atlas — MODIS & VIIRS sıcak nokta harmonizasyon motoru.

Boru hattı (pipeline):
    Aşama 1  NASA FIRMS verisi -> ortak şema, UTC zaman damgası, güven filtresi
    Aşama 2  Haversine mekansal-zamansal kümeleme -> Harmonize Yangın Olayı (Unified Hotspot ID)
    Aşama 3  FRP çapraz kalibrasyonu (VIIRS -> MODIS eşdeğeri), katsayılar veriden öğrenilir
    Aşama 4  Statik termal kaynak (gaz bacası / sanayi) tespiti, mekansal-zamansal kalıcılık

Takım: Kirvecanlar — 2026 NASA Space Apps Challenge, Harmonization of MODIS and VIIRS Hot Spots
"""
from __future__ import annotations

import calendar
import io
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0
KM_PER_DEG_LAT = 111.32

# FIRMS Area API bounding box format: west,south,east,north
AOI_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "Turkey (National)": (25.6, 35.8, 44.8, 42.1),
    "Southeastern Anatolia & Batman Basin": (38.5, 36.6, 43.5, 38.8),
    "Mediterranean & Aegean Forest Belt": (26.0, 36.0, 36.5, 40.0),
    # Backwards compatibility aliases
    "Türkiye Geneli": (25.6, 35.8, 44.8, 42.1),
    "Güneydoğu Anadolu & Batman Havzası": (38.5, 36.6, 43.5, 38.8),
    "Akdeniz & Ege Orman Kuşağı": (26.0, 36.0, 36.5, 40.0),
}

FIRMS_SOURCES = [
    "MODIS_NRT",
    "MODIS_SP",
    "VIIRS_SNPP_NRT",
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_NRT",
]

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{day_range}/{date}"
FIRMS_MAX_DAY_RANGE = 5  # NASA FIRMS Area API tek istekte en fazla 5 gün kabul eder [1..5]

CONF_LEVELS = {"low": 0, "nominal": 1, "high": 2}
CONF_NAMES = {v: k for k, v in CONF_LEVELS.items()}

SENSOR_BOTH = "BOTH (MODIS+VIIRS)"
SENSOR_VIIRS = "VIIRS_CLUSTER_HARMONIZED"
SENSOR_MODIS = "MODIS_ONLY"


# ---------------------------------------------------------------------------
# Geometri
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vektörize Haversine büyük çember mesafesi (km), R = 6371.0 km."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _unit_sphere_xyz(lat, lon) -> np.ndarray:
    phi, lmb = np.radians(lat), np.radians(lon)
    c_phi = np.cos(phi)
    return np.column_stack([c_phi * np.cos(lmb), c_phi * np.sin(lmb), np.sin(phi)])


def _grid_keys(lat: np.ndarray, lon: np.ndarray, cell_km: float, offset: float = 0.0) -> np.ndarray:
    """~cell_km × cell_km grid hücre anahtarı (boylam adımı enleme göre cos düzeltmeli)."""
    dlat = cell_km / KM_PER_DEG_LAT
    iy = np.floor(lat / dlat + offset).astype(np.int64)
    row_center = (iy - offset + 0.5) * dlat
    dlon = cell_km / (KM_PER_DEG_LAT * np.cos(np.radians(row_center)))
    ix = np.floor(lon / dlon + offset).astype(np.int64)
    return iy * 10_000_000 + ix


def _spatial_components(lat: np.ndarray, lon: np.ndarray, distance_km: float) -> np.ndarray:
    """
    Yalnızca mekansal Haversine (Δd ≤ distance_km) tek bağlantılı kümeleme (statik kaynak birleştirme).
    Köşegeni ≤ distance_km olan mikro-hücrelerdeki noktalar garanti olarak birbirine bağlıdır;
    önce mikro-hücre bazında indirgeme yapılarak yoğun kümelerdeki O(N²) çift patlaması önlenir.
    """
    n = len(lat)
    if n == 0:
        return np.array([], dtype=int)
    micro_km = distance_km / 1.5  # mikro-hücre köşegeni < distance_km
    m_keys = _grid_keys(lat, lon, micro_km, 0.0)
    uniq_m, first_idx, m_inv = np.unique(m_keys, return_index=True, return_inverse=True)
    n_m = len(uniq_m)
    if n_m == 1:
        return np.zeros(n, dtype=int)

    rep_lat, rep_lon = lat[first_idx], lon[first_idx]
    xyz = _unit_sphere_xyz(rep_lat, rep_lon)
    chord_max = 2.0 * np.sin((distance_km * 2.5) / (2.0 * EARTH_RADIUS_KM))
    cand_pairs = cKDTree(xyz).query_pairs(r=chord_max, output_type="ndarray")

    if len(cand_pairs) == 0:
        first_appear = np.full(n_m, n + 1, dtype=int)
        np.minimum.at(first_appear, m_inv, np.arange(n))
        rank = np.argsort(np.argsort(first_appear))
        return rank[m_inv]

    order = np.argsort(m_inv, kind="stable")
    sorted_m = m_inv[order]
    splits = np.searchsorted(sorted_m, np.arange(n_m))
    splits_end = np.append(splits[1:], n)
    chord_exact = 2.0 * np.sin(distance_km / (2.0 * EARTH_RADIUS_KM)) * (1 + 1e-9)

    edges_u, edges_v = [], []
    for u, v in cand_pairs:
        if haversine_km(rep_lat[u], rep_lon[u], rep_lat[v], rep_lon[v]) <= distance_km:
            edges_u.append(u)
            edges_v.append(v)
            continue
        idx_u = order[splits[u]:splits_end[u]]
        idx_v = order[splits[v]:splits_end[v]]
        xyz_u = _unit_sphere_xyz(lat[idx_u], lon[idx_u])
        xyz_v = _unit_sphere_xyz(lat[idx_v], lon[idx_v])
        hit = cKDTree(xyz_u).query_ball_tree(cKDTree(xyz_v), r=chord_exact)
        for ii, nbrs in enumerate(hit):
            if nbrs:
                jj = idx_v[nbrs]
                if np.any(haversine_km(lat[idx_u[ii]], lon[idx_u[ii]], lat[jj], lon[jj]) <= distance_km):
                    edges_u.append(u)
                    edges_v.append(v)
                    break

    graph = coo_matrix((np.ones(len(edges_u), dtype=np.int8), (edges_u, edges_v)), shape=(n_m, n_m))
    _, comp_m = connected_components(graph, directed=False)
    pt_comp = comp_m[m_inv]
    first_appear = np.full(int(pt_comp.max()) + 1, n + 1, dtype=int)
    np.minimum.at(first_appear, pt_comp, np.arange(n))
    rank = np.argsort(np.argsort(first_appear))
    return rank[pt_comp]


# ---------------------------------------------------------------------------
# Aşama 1 — Veri çekme ve şema standardizasyonu
# ---------------------------------------------------------------------------
class FIRMSClient:
    """NASA FIRMS Area API istemcisi (CSV, paralel HTTP çekimi destekli)."""

    def __init__(self, map_key: str, timeout: int = 60):
        if not map_key or not map_key.strip():
            raise ValueError("FIRMS MAP_KEY boş olamaz.")
        self.map_key = map_key.strip()
        self.timeout = timeout

    @staticmethod
    def area_string(bbox: tuple[float, float, float, float]) -> str:
        w, s, e, n = bbox
        return f"{w},{s},{e},{n}"

    def build_url(self, source: str, bbox, day_range: int, start: date) -> str:
        return FIRMS_AREA_URL.format(
            key=self.map_key,
            source=source,
            area=self.area_string(bbox),
            day_range=int(day_range),
            date=start.isoformat(),
        )

    def request_plan(self, source: str, bbox, start: date, end: date) -> list[str]:
        """[start, end] aralığını 5 günlük parçalara bölüp URL listesi döndürür."""
        urls, cur = [], start
        while cur <= end:
            span = min(FIRMS_MAX_DAY_RANGE, (end - cur).days + 1)
            urls.append(self.build_url(source, bbox, span, cur))
            cur += timedelta(days=span)
        return urls

    def _get(self, url: str) -> pd.DataFrame:
        req = urllib.request.Request(url, headers={"User-Agent": "KOZ-Atlas/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            err_msg = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"HTTP {exc.code}: {err_msg[:200]}") from exc
        if not text.lstrip().lower().startswith("latitude"):
            raise RuntimeError(f"FIRMS yanıtı CSV değil: {text.strip()[:200]}")
        return pd.read_csv(io.StringIO(text))

    def fetch(self, sources: list[str], bbox, start: date, end: date) -> tuple[pd.DataFrame, list[str]]:
        tasks: list[tuple[int, str, str]] = []
        idx = 0
        for src in sources:
            for url in self.request_plan(src, bbox, start, end):
                tasks.append((idx, src, url))
                idx += 1
        if not tasks:
            return pd.DataFrame(), []

        results: dict[int, pd.DataFrame] = {}
        err_map: dict[int, str] = {}
        max_workers = min(8, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut_to_task = {pool.submit(self._get, url): (i, src) for i, src, url in tasks}
            for fut in as_completed(fut_to_task):
                i, src = fut_to_task[fut]
                try:
                    df = fut.result()
                    if len(df):
                        df["source"] = src
                        results[i] = df
                except (urllib.error.URLError, RuntimeError, pd.errors.ParserError, TimeoutError) as exc:
                    err_map[i] = f"{src}: {exc}"

        errors = [err_map[k] for k in sorted(err_map)]
        frames = [results[k] for k in sorted(results)]
        if not frames:
            return pd.DataFrame(), errors
        return pd.concat(frames, ignore_index=True), errors


def _infer_instrument(df: pd.DataFrame) -> pd.Series:
    if "source" in df.columns:
        fallback = pd.Series(
            np.where(df["source"].astype(str).str.upper().str.contains("VIIRS"), "VIIRS", "MODIS"), index=df.index
        )
    else:
        fallback = pd.Series("VIIRS" if "bright_ti4" in df.columns else "MODIS", index=df.index)
    if "instrument" in df.columns:
        inst = df["instrument"].astype(str).str.upper().str.strip()
        return inst.where(inst.isin(["MODIS", "VIIRS"]), fallback)
    return fallback


def _numeric_conf_level(num: pd.Series) -> np.ndarray:
    # FIRMS MODIS sınıflaması: 0–29 low, 30–79 nominal, 80–100 high (sayı yoksa nominal)
    return np.select([num.isna(), num < 30, num < 80], [1, 0, 1], default=2)


def _confidence_level(conf: pd.Series, instrument: pd.Series) -> pd.Series:
    """
    Ortak güven ölçeği: 0 = low, 1 = nominal, 2 = high.
    MODIS: 0–100 sayısal güven; VIIRS: l / n / h kategorik güven.
    """
    level = pd.Series(np.nan, index=conf.index, dtype="float64")
    is_modis = (instrument == "MODIS").to_numpy()
    if is_modis.any():
        level[is_modis] = _numeric_conf_level(pd.to_numeric(conf[is_modis], errors="coerce"))
    if (~is_modis).any():
        txt = conf[~is_modis].astype(str).str.lower().str.strip().str[:1]
        level[~is_modis] = txt.map({"l": 0, "n": 1, "h": 2})
    leftover = level.isna()
    if leftover.any():  # VIIRS satırında sayısal güven gelirse sayısal ölçek uygulanır
        level[leftover] = _numeric_conf_level(pd.to_numeric(conf[leftover], errors="coerce"))
    return level.astype(int)


def standardize_firms(df: pd.DataFrame) -> pd.DataFrame:
    """FIRMS MODIS ve VIIRS CSV'lerini tek ortak şemaya dönüştürür."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.rename(columns=lambda c: str(c).strip().lower())
    required = {"latitude", "longitude", "acq_date", "acq_time"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Eksik zorunlu FIRMS sütunları: {sorted(missing)}")

    def num(col):
        if col not in d:
            return pd.Series(np.nan, index=d.index, dtype="float64")
        s = d[col]
        if pd.api.types.is_numeric_dtype(s):
            return s.astype("float64", copy=False)
        return pd.to_numeric(s, errors="coerce")

    out = pd.DataFrame(index=d.index)
    out["latitude"] = num("latitude")
    out["longitude"] = num("longitude")
    out["instrument"] = _infer_instrument(d).astype(str)
    is_viirs = (out["instrument"] == "VIIRS").to_numpy()

    out["brightness_k"] = np.where(is_viirs, num("bright_ti4"), num("brightness"))
    out["brightness_2_k"] = np.where(is_viirs, num("bright_ti5"), num("bright_t31"))
    out["scan"] = num("scan")
    out["track"] = num("track")
    out["frp"] = num("frp").fillna(0.0).clip(lower=0.0)

    hhmm = num("acq_time").fillna(0).astype(int).clip(0, 2359)
    acq_dt = pd.to_datetime(d["acq_date"], errors="coerce")
    total_mins = (hhmm // 100) * 60 + (hhmm % 100)
    out["utc_timestamp"] = acq_dt + pd.to_timedelta(total_mins, unit="m")
    out["acq_date"] = acq_dt.dt.normalize()

    out["satellite"] = d["satellite"].astype(str) if "satellite" in d else out["instrument"]
    out["confidence_raw"] = d["confidence"].astype(str) if "confidence" in d else ""
    if "confidence" in d:
        out["conf_level"] = _confidence_level(d["confidence"], out["instrument"])
    else:
        out["conf_level"] = 1
    out["conf_class"] = out["conf_level"].map(CONF_NAMES)
    out["daynight"] = d["daynight"].astype(str) if "daynight" in d else ""
    out["version"] = d["version"].astype(str) if "version" in d else ""
    out["source"] = d["source"].astype(str) if "source" in d else out["instrument"]

    valid = out["latitude"].notna() & out["longitude"].notna() & out["utc_timestamp"].notna()
    if not valid.all():
        out = out[valid]
    out = out.sort_values("utc_timestamp", kind="stable").reset_index(drop=True)
    out["point_id"] = np.arange(len(out))
    return out


def filter_aoi(df: pd.DataFrame, bbox) -> pd.DataFrame:
    w, s, e, n = bbox
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    m = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
    if m.all():
        return df.reset_index(drop=True)
    return df[m].reset_index(drop=True)


def filter_dates(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    ad = df["acq_date"]
    m = (ad >= t0) & (ad <= t1)
    if m.all():
        return df.reset_index(drop=True)
    return df[m].reset_index(drop=True)


def filter_confidence(df: pd.DataFrame, min_level: int) -> pd.DataFrame:
    if int(min_level) <= 0:
        return df.reset_index(drop=True)
    m = df["conf_level"].to_numpy() >= int(min_level)
    if m.all():
        return df.reset_index(drop=True)
    return df[m].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aşama 2 — Haversine mekansal-zamansal kümeleme
# ---------------------------------------------------------------------------
def spatiotemporal_labels(df: pd.DataFrame, distance_km: float, window_min: float) -> np.ndarray:
    """
    Δd ≤ distance_km (Haversine) ve |Δt| ≤ window_min koşulunu birlikte sağlayan nokta çiftlerini
    kenar kabul eden grafiğin bağlı bileşenlerini (tek bağlantılı kümeleme) döndürür.

    Aday çiftler 4 boyutlu bir KD-ağacında aranır: birim küre üzerindeki (x, y, z) ve
    kiriş eşiğine ölçeklenmiş zaman ekseni. İki koşulu sağlayan her çift bu uzayda
    √2·kiriş yarıçapı içinde kalır; adaylar ardından tam Haversine ve |Δt| ile doğrulanır.
    """
    n = len(df)
    if n == 0:
        return np.array([], dtype=int)
    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    t_min = (df["utc_timestamp"] - pd.Timestamp("2000-01-01")).dt.total_seconds().to_numpy() / 60.0

    chord = 2.0 * np.sin(distance_km / (2.0 * EARTH_RADIUS_KM))
    scale = chord / max(window_min, 1e-6)
    pts = np.column_stack([_unit_sphere_xyz(lat, lon), t_min * scale])
    pairs = cKDTree(pts).query_pairs(r=chord * np.sqrt(2.0) * (1 + 1e-9), output_type="ndarray")

    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        dt_ok = np.abs(t_min[i] - t_min[j]) <= window_min
        i, j = i[dt_ok], j[dt_ok]
        if len(i):
            dist_ok = haversine_km(lat[i], lon[i], lat[j], lon[j]) <= distance_km
            i, j = i[dist_ok], j[dist_ok]
    else:
        i = j = np.array([], dtype=int)

    graph = coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def build_events(points: pd.DataFrame) -> pd.DataFrame:
    """Kümelenmiş noktalardan Harmonize Yangın Olayı tablosu üretir (vektörize ve optimize)."""
    p = points
    w = p["frp"].to_numpy() + 1e-6  # FRP=0 olan kümelerde eşit ağırlık
    p_agg = p.assign(
        _w=w,
        _wlat=w * p["latitude"].to_numpy(),
        _wlon=w * p["longitude"].to_numpy(),
        _is_modis=(p["instrument"] == "MODIS").astype(int),
        _is_viirs=(p["instrument"] == "VIIRS").astype(int),
    )
    g = p_agg.groupby("event_idx", sort=False)
    ev = g.agg(
        _wlat=("_wlat", "sum"),
        _wlon=("_wlon", "sum"),
        _w=("_w", "sum"),
        n_modis=("_is_modis", "sum"),
        n_viirs=("_is_viirs", "sum"),
        utc_timestamp=("utc_timestamp", "min"),
        utc_end=("utc_timestamp", "max"),
        frp_max_pixel=("frp", "max"),
        conf_level=("conf_level", "max"),
        brightness_max_k=("brightness_k", "max"),
        daynight=("daynight", "first"),
    )
    ev["latitude"] = ev["_wlat"] / ev["_w"]
    ev["longitude"] = ev["_wlon"] / ev["_w"]
    ev = ev.drop(columns=["_wlat", "_wlon", "_w"])

    # Fast bitmask lookup for comma-sorted satellite lists per event
    sats_cat = pd.Categorical(p["satellite"].fillna(""))
    uniq_sats = [str(s) for s in sats_cat.categories]
    if len(uniq_sats) <= 16:
        order = np.argsort(uniq_sats)
        sorted_sats = [uniq_sats[i] for i in order]
        rank_of_code = np.empty(len(uniq_sats), dtype=int)
        rank_of_code[order] = np.arange(len(uniq_sats))
        bits = 1 << rank_of_code[sats_cat.codes]
        n_ev = int(p["event_idx"].max()) + 1
        masks = np.zeros(n_ev, dtype=int)
        np.bitwise_or.at(masks, p["event_idx"].to_numpy(), bits)
        lookup = np.array(
            [
                ",".join(sorted_sats[b] for b in range(len(sorted_sats)) if (m & (1 << b)) and sorted_sats[b])
                for m in range(1 << len(sorted_sats))
            ],
            dtype=object,
        )
        ev["satellites"] = lookup[masks[ev.index.to_numpy()]]
    else:
        dedup_sat = p[["event_idx", "satellite"]].drop_duplicates().sort_values(["event_idx", "satellite"])
        ev["satellites"] = dedup_sat.groupby("event_idx", sort=False)["satellite"].agg(",".join)

    # Geçiş bazında FRP toplamı, ardından sensör bazında ortalama
    per_pass = p.groupby(["event_idx", "instrument", "satellite"], sort=False, as_index=False)["frp"].sum()
    per_sensor = per_pass.groupby(["event_idx", "instrument"], sort=False)["frp"].mean().unstack("instrument", fill_value=0.0)
    ev["frp_modis_mw"] = per_sensor.get("MODIS", pd.Series(0.0, index=ev.index)).reindex(ev.index).fillna(0.0)
    ev["frp_viirs_mw"] = per_sensor.get("VIIRS", pd.Series(0.0, index=ev.index)).reindex(ev.index).fillna(0.0)

    ev["merged_pixel_count"] = ev["n_modis"] + ev["n_viirs"]
    ev["sensor_label"] = np.select(
        [(ev["n_modis"] > 0) & (ev["n_viirs"] > 0), ev["n_viirs"] > 0],
        [SENSOR_BOTH, SENSOR_VIIRS],
        default=SENSOR_MODIS,
    )
    ev["conf_class"] = ev["conf_level"].map(CONF_NAMES)
    ev["acq_date"] = ev["utc_timestamp"].dt.normalize()
    ev = ev.sort_values("utc_timestamp", kind="stable").reset_index()
    seq = ev.groupby("acq_date", sort=False).cumcount() + 1
    ymd = (ev["acq_date"].dt.year * 10000 + ev["acq_date"].dt.month * 100 + ev["acq_date"].dt.day).astype(str)
    ev["unified_hotspot_id"] = "KOZ-" + ymd + "-" + seq.astype(str).str.zfill(5)
    return ev


# ---------------------------------------------------------------------------
# Aşama 3 — FRP çapraz kalibrasyonu ve MODIS eşdeğerliği
# ---------------------------------------------------------------------------
@dataclass
class FRPCalibration:
    method: str  # "loglog" | "linear" | "identity"
    a: float  # loglog: log10 kesişim; linear: eğim
    b: float  # loglog: üs
    n_pairs: int
    r: float
    note: str

    def apply(self, viirs_frp) -> np.ndarray:
        x = np.asarray(viirs_frp, dtype=float)
        if self.method == "loglog":
            return np.where(x > 0, 10 ** self.a * np.power(np.clip(x, 1e-9, None), self.b), 0.0)
        if self.method == "linear":
            return self.a * x
        return x

    def describe(self) -> str:
        if self.method == "loglog":
            return f"FRP_MODIS ≈ {10 ** self.a:.3f} · FRP_VIIRS^{self.b:.3f}   (n={self.n_pairs}, r={self.r:.3f})"
        if self.method == "linear":
            return f"FRP_MODIS ≈ {self.a:.3f} · FRP_VIIRS   (n={self.n_pairs}, r={self.r:.3f})"
        return f"Kimlik dönüşümü (FRP_MODIS = FRP_VIIRS) — {self.note}"


def fit_frp_calibration(events: pd.DataFrame, method: str = "loglog", min_pairs: int = 10) -> FRPCalibration:
    """
    Aynı (Δd, Δt) zarfında hem MODIS hem VIIRS tarafından görülen olaylardan (BOTH)
    VIIRS -> MODIS FRP dönüşümünü öğrenir. Yeterli eş-konumlu çift yoksa kimlik dönüşümü kullanılır.
        loglog : log10(FRP_MODIS) = a + b·log10(FRP_VIIRS)   (EKK)
        linear : FRP_MODIS = a·FRP_VIIRS                      (orijinden geçen EKK)
    """
    both = events[(events["frp_modis_mw"] > 0) & (events["frp_viirs_mw"] > 0)]
    n = len(both)
    if method == "identity" or n < min_pairs:
        note = "kullanıcı seçimi" if method == "identity" else f"yetersiz eş-konumlu çift (n={n} < {min_pairs})"
        return FRPCalibration("identity", 1.0, 1.0, n, float("nan"), note)
    x = both["frp_viirs_mw"].to_numpy(float)
    y = both["frp_modis_mw"].to_numpy(float)
    if method == "linear":
        slope = float(np.sum(x * y) / np.sum(x * x))
        return FRPCalibration("linear", slope, 0.0, n, float(np.corrcoef(x, y)[0, 1]), "orijinden geçen EKK")
    res = stats.linregress(np.log10(x), np.log10(y))
    return FRPCalibration("loglog", float(res.intercept), float(res.slope), n, float(res.rvalue), "log10-log10 EKK")


def apply_frp_calibration(events: pd.DataFrame, cal: FRPCalibration) -> pd.DataFrame:
    """
    calibrated_frp_mw:
        MODIS_ONLY / BOTH -> gözlenen MODIS FRP (referans sensör, 2000'den beri kesintisiz seri)
        VIIRS-only        -> kalibrasyon fonksiyonu ile MODIS eşdeğeri
    """
    ev = events.copy()
    ev["frp_viirs_calibrated_mw"] = cal.apply(ev["frp_viirs_mw"].to_numpy())
    has_modis = ev["n_modis"] > 0
    ev["calibrated_frp_mw"] = np.where(has_modis, ev["frp_modis_mw"], ev["frp_viirs_calibrated_mw"])
    return ev


@dataclass
class DetectionCurve:
    log_frp_centers: np.ndarray
    p_detect: np.ndarray
    raw_ratio: np.ndarray
    plateau: float
    n_events: int

    @property
    def available(self) -> bool:
        return len(self.log_frp_centers) > 0

    def weight(self, frp_mw) -> np.ndarray:
        frp_mw = np.asarray(frp_mw, float)
        if not self.available:
            return np.ones(len(frp_mw))
        x = np.log10(np.clip(frp_mw, 1e-3, None))
        return np.interp(x, self.log_frp_centers, self.p_detect)


def estimate_modis_detection_curve(events: pd.DataFrame, n_bins: int = 12, min_events: int = 200) -> DetectionCurve:
    """
    VIIRS'in gördüğü dinamik olaylar içinde MODIS'in de aynı zarfta tespit ettiği olayların oranı,
    kalibre FRP sınıflarına göre hesaplanır. Yüksek FRP sınıflarında oran bir platoya ulaşır
    (MODIS'in o pencerede gözlem yapma olasılığı); oran bu platoya bölünerek MODIS'in FRP'ye bağlı
    göreli tespit olasılığı elde edilir ve monoton artan hale getirilir. Sabit eşik kullanılmaz.
    """
    empty = DetectionCurve(np.array([]), np.array([]), np.array([]), float("nan"), 0)
    v = events[(events["n_viirs"] > 0) & (events["frp_viirs_calibrated_mw"] > 0)]
    if "is_static_source" in v:
        v = v[~v["is_static_source"]]
    if len(v) < min_events or (v["n_modis"] > 0).sum() < 20:
        empty.n_events = len(v)
        return empty
    x = np.log10(v["frp_viirs_calibrated_mw"].to_numpy(float))
    hit = (v["n_modis"] > 0).to_numpy(float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    nb = len(edges) - 1
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nb - 1)
    centers = np.array([x[idx == b].mean() for b in range(nb)])
    ratio = np.array([hit[idx == b].mean() for b in range(nb)])
    plateau = float(np.mean(np.sort(ratio)[-3:]))
    p = np.maximum.accumulate(np.clip(ratio / plateau, 0.0, 1.0)) if plateau > 0 else np.ones(nb)
    return DetectionCurve(centers, p, ratio, plateau, len(v))


def apply_modis_equivalence(events: pd.DataFrame, curve: DetectionCurve) -> pd.DataFrame:
    """
    modis_eq_weight : olayın MODIS tarafından tespit edilme olasılığı (MODIS gözlemi varsa 1.0).
    modis_equivalent: modis_eq_weight ≥ 0.5.
    """
    ev = events.copy()
    w = curve.weight(ev["calibrated_frp_mw"].to_numpy())
    ev["modis_eq_weight"] = np.where(ev["n_modis"] > 0, 1.0, w).round(4)
    ev["modis_equivalent"] = ev["modis_eq_weight"] >= 0.5
    return ev


# ---------------------------------------------------------------------------
# Aşama 4 — Statik termal kaynak tespiti
# ---------------------------------------------------------------------------
def detect_static_sources(
    events: pd.DataFrame, cell_km: float = 1.0, min_days: int = 5, min_months: int = 4
) -> tuple[pd.Series, pd.DataFrame]:
    """
    ~cell_km boyutlu grid hücrelerinde olayların zamansal kalıcılığını ölçer.
    Statik kaynak koşulu (hücre-yıl bazında): aynı takvim yılı içinde ≥ min_days farklı günde olay
    VE bu olayların ≥ min_months farklı aya yayılması. Mevsimsel biyokütle yangınları (anız, orman)
    bir hücrede yıl içinde birkaç güne/aya yoğunlaşırken gaz bacaları ve sanayi tesisleri yıl boyu
    sinyal verir. Herhangi bir yılda koşulu sağlayan hücre tüm dönem için statik işaretlenir.
    Hücre sınırı etkisini azaltmak için yarım hücre kaydırılmış ikinci grid de değerlendirilir.
    """
    if events.empty:
        return pd.Series(False, index=events.index), pd.DataFrame()
    lat = events["latitude"].to_numpy(float)
    lon = events["longitude"].to_numpy(float)
    day = events["acq_date"].to_numpy()
    month = events["acq_date"].dt.month.to_numpy()
    year = events["acq_date"].dt.year.to_numpy()

    flag = np.zeros(len(events), dtype=bool)
    days_best = np.zeros(len(events), dtype=int)
    months_best = np.zeros(len(events), dtype=int)
    for offset in (0.0, 0.5):
        key = _grid_keys(lat, lon, cell_km, offset)
        sub = pd.DataFrame({"key": key, "year": year, "day": day, "month": month})
        st_df = sub.groupby(["key", "year"], sort=False).agg(nd=("day", "nunique"), nm=("month", "nunique")).reset_index()
        is_stat_year = (st_df["nd"] >= min_days) & (st_df["nm"] >= min_months)
        stat_keys = st_df.loc[is_stat_year, "key"].to_numpy()
        key_max = st_df.groupby("key", sort=False).agg(nd=("nd", "max"), nm=("nm", "max"))

        k_idx = pd.Index(key)
        flag |= k_idx.isin(stat_keys)
        days_best = np.maximum(days_best, key_max["nd"].reindex(k_idx, fill_value=0).to_numpy())
        months_best = np.maximum(months_best, key_max["nm"].reindex(k_idx, fill_value=0).to_numpy())

    is_static = pd.Series(flag, index=events.index)
    st = events.assign(persistence_days=days_best, persistence_months=months_best)[flag]
    if st.empty:
        return is_static, pd.DataFrame()
    # Komşu statik hücreleri tek tesis olarak birleştir (mikro-hücre hızlandırmalı mekansal bağlantı)
    site_lbl = _spatial_components(
        st["latitude"].to_numpy(float), st["longitude"].to_numpy(float), distance_km=cell_km * 1.5
    )
    sites = (
        st.assign(site=site_lbl)
        .groupby("site")
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            event_count=("unified_hotspot_id", "size"),
            active_days=("acq_date", "nunique"),
            max_days_in_year=("persistence_days", "max"),
            max_months_in_year=("persistence_months", "max"),
            first_seen=("acq_date", "min"),
            last_seen=("acq_date", "max"),
            mean_calibrated_frp_mw=("calibrated_frp_mw", "mean"),
        )
        .sort_values("event_count", ascending=False)
        .reset_index(drop=True)
    )
    sites.insert(0, "static_site_id", [f"STK-{i + 1:03d}" for i in range(len(sites))])
    return is_static, sites


# ---------------------------------------------------------------------------
# Uçtan uca boru hattı
# ---------------------------------------------------------------------------
@dataclass
class PipelineParams:
    distance_km: float = 1.2
    window_min: float = 60.0
    min_conf_level: int = 1  # 0 low, 1 nominal, 2 high
    calibration_method: str = "loglog"
    static_cell_km: float = 1.0
    static_min_days: int = 5
    static_min_months: int = 4


@dataclass
class PipelineResult:
    points: pd.DataFrame
    events: pd.DataFrame
    static_sites: pd.DataFrame
    calibration: FRPCalibration
    detection_curve: DetectionCurve
    n_raw_input: int
    n_after_confidence: int
    params: PipelineParams
    stats: dict = field(default_factory=dict)


def run_pipeline(std_points: pd.DataFrame, params: PipelineParams) -> PipelineResult:
    n_raw = len(std_points)
    pts = filter_confidence(std_points, params.min_conf_level) if n_raw else std_points
    if pts.empty:
        empty = pd.DataFrame()
        cal = FRPCalibration("identity", 1.0, 1.0, 0, float("nan"), "veri yok")
        curve = DetectionCurve(np.array([]), np.array([]), np.array([]), float("nan"), 0)
        return PipelineResult(pts, empty, empty, cal, curve, n_raw, 0, params, {})

    # Aşama 2
    pts = pts.copy()
    pts["event_idx"] = spatiotemporal_labels(pts, params.distance_km, params.window_min)
    events = build_events(pts)

    # Aşama 3
    cal = fit_frp_calibration(events, params.calibration_method)
    events = apply_frp_calibration(events, cal)

    # Aşama 4
    is_static, sites = detect_static_sources(
        events, params.static_cell_km, params.static_min_days, params.static_min_months
    )
    events["is_static_source"] = is_static.to_numpy()
    events["event_class"] = np.where(
        events["is_static_source"], "Static Industrial Thermal Source", "Dynamic Biomass Fire"
    )

    # MODIS eşdeğerliği (kalibre FRP + statik maske ile)
    curve = estimate_modis_detection_curve(events)
    events = apply_modis_equivalence(events, curve)

    # Doğrudan dizi indeksleme ile hızlı eşleştirme
    n_ev = len(events)
    ev_idx_arr = events["event_idx"].to_numpy()
    id_lookup = np.empty(n_ev, dtype=object)
    stat_lookup = np.empty(n_ev, dtype=bool)
    id_lookup[ev_idx_arr] = events["unified_hotspot_id"].to_numpy()
    stat_lookup[ev_idx_arr] = events["is_static_source"].to_numpy()

    pts_eidx = pts["event_idx"].to_numpy()
    pts["unified_hotspot_id"] = id_lookup[pts_eidx]
    pts["is_static_source"] = stat_lookup[pts_eidx]

    n_pts = len(pts)
    dyn = ~events["is_static_source"]
    inst_arr = pts["instrument"].to_numpy()
    raw_modis = int((inst_arr == "MODIS").sum())
    stats_ = {
        "raw_points": n_pts,
        "raw_modis": raw_modis,
        "raw_viirs": n_pts - raw_modis,
        "events": n_ev,
        "dedup_ratio": 1.0 - n_ev / n_pts if n_pts else 0.0,
        "static_events": int(events["is_static_source"].sum()),
        "static_sites": len(sites),
        "mean_calibrated_frp": float(events.loc[dyn, "calibrated_frp_mw"].mean()) if dyn.any() else float("nan"),
        "label_counts": events["sensor_label"].value_counts().to_dict(),
    }
    return PipelineResult(pts, events, sites, cal, curve, n_raw, n_pts, params, stats_)


# ---------------------------------------------------------------------------
# Yanma Aktivitesi Takvimi
# ---------------------------------------------------------------------------
CALENDAR_METRICS = {
    "celldays": "MODIS-equivalent active fire cell-days",
    "events": "KOZ harmonized event count",
    "frp": "Total calibrated FRP (MW)",
}


def cell_day_table(events: pd.DataFrame, cell_km: float = 1.0) -> pd.DataFrame:
    """
    Olayları ~cell_km grid hücresi × UTC gün düzeyinde toplar. Aynı gün içindeki farklı uydu geçişleri
    aynı yangını tekrar saymaz; uydu sayısındaki değişimden bağımsız 'aktif yangın hücre-günü' üretir.
    Hücre-gün ağırlığı = o hücre-gündeki olayların en yüksek modis_eq_weight değeri.
    """
    if events.empty:
        return pd.DataFrame()
    key = _grid_keys(events["latitude"].to_numpy(float), events["longitude"].to_numpy(float), cell_km)
    return (
        events.assign(cell=key)
        .groupby(["cell", "acq_date"])
        .agg(
            modis_eq_weight=("modis_eq_weight", "max"),
            n_events=("unified_hotspot_id", "size"),
            calibrated_frp_mw=("calibrated_frp_mw", "max"),
            is_static_source=("is_static_source", "max"),
        )
        .reset_index()
    )


def monthly_matrix(events: pd.DataFrame, metric: str = "celldays", cell_km: float = 1.0) -> pd.DataFrame:
    """Yıl × Ay matrisi."""
    if events.empty:
        return pd.DataFrame()
    if metric == "celldays":
        base, col, agg = cell_day_table(events, cell_km), "modis_eq_weight", "sum"
    elif metric == "frp":
        base, col, agg = events, "calibrated_frp_mw", "sum"
    else:
        base, col, agg = events, "unified_hotspot_id", "count"
    e = base.assign(year=base["acq_date"].dt.year, month=base["acq_date"].dt.month)
    m = e.pivot_table(index="year", columns="month", values=col, aggfunc=agg, fill_value=0)
    years = range(int(e["year"].min()), int(e["year"].max()) + 1)
    return m.reindex(index=years, columns=range(1, 13), fill_value=0).astype(float)


def month_anomaly(matrix: pd.DataFrame, year: int, month: int) -> dict:
    """Seçilen ay değerini, aynı takvim ayının diğer yıllardaki tarihsel medyanıyla karşılaştırır."""
    if matrix.empty or year not in matrix.index:
        return {}
    current = float(matrix.loc[year, month])
    hist = matrix.loc[matrix.index != year, month].astype(float)
    if hist.empty:
        nan = float("nan")
        return {"current": current, "median": nan, "q25": nan, "q75": nan, "ratio": nan, "percentile": nan, "n_years": 0}
    med = float(hist.median())
    return {
        "current": current,
        "median": med,
        "q25": float(hist.quantile(0.25)),
        "q75": float(hist.quantile(0.75)),
        "ratio": current / med if med > 0 else float("nan"),
        "percentile": float(stats.percentileofscore(hist, current, kind="weak")),
        "n_years": int(len(hist)),
    }


def annual_series(points: pd.DataFrame, events: pd.DataFrame, cell_km: float = 1.0) -> pd.DataFrame:
    """Annual time series before and after harmonization."""
    if events.empty:
        return pd.DataFrame()
    py = points["acq_date"].dt.year
    ey = events["acq_date"].dt.year
    cd = cell_day_table(events, cell_km)
    inst_arr = points["instrument"].to_numpy()
    df = pd.DataFrame(
        {
            "Raw MODIS Pixels": py[inst_arr == "MODIS"].value_counts(),
            "Raw VIIRS Pixels": py[inst_arr == "VIIRS"].value_counts(),
            "KOZ Harmonized Events": ey.value_counts(),
            "KOZ MODIS-Equivalent Cell-Days": cd.groupby(cd["acq_date"].dt.year)["modis_eq_weight"].sum(),
        }
    )
    years = range(int(ey.min()), int(ey.max()) + 1)
    df = df.reindex(years).fillna(0).round(0).astype(int)
    df.insert(0, "Raw Total (MODIS+VIIRS)", df["Raw MODIS Pixels"] + df["Raw VIIRS Pixels"])

    # Backwards-compatibility aliases for Turkish callers
    df["Ham MODIS piksel"] = df["Raw MODIS Pixels"]
    df["Ham VIIRS piksel"] = df["Raw VIIRS Pixels"]
    df["KÖZ harmonize olay"] = df["KOZ Harmonized Events"]
    df["KÖZ MODIS-eşdeğer hücre-günü"] = df["KOZ MODIS-Equivalent Cell-Days"]
    df["Ham toplam (MODIS+VIIRS)"] = df["Raw Total (MODIS+VIIRS)"]

    df.index.name = "year"
    return df


def break_ratio(series: pd.Series, break_year: int = 2012) -> float:
    """(break_year ve sonrası ortalama) / (öncesi ortalama). 1.0'a yakın değer = sensör kırılması yok."""
    pre = series[series.index < break_year]
    post = series[series.index >= break_year]
    if pre.empty or post.empty or pre.mean() == 0:
        return float("nan")
    return float(post.mean() / pre.mean())


# ---------------------------------------------------------------------------
# Dışa aktarma
# ---------------------------------------------------------------------------
EXPORT_COLUMNS = [
    "unified_hotspot_id",
    "latitude",
    "longitude",
    "utc_timestamp",
    "utc_end",
    "acq_date",
    "sensor_label",
    "merged_pixel_count",
    "n_modis",
    "n_viirs",
    "satellites",
    "frp_modis_mw",
    "frp_viirs_mw",
    "frp_viirs_calibrated_mw",
    "calibrated_frp_mw",
    "modis_eq_weight",
    "modis_equivalent",
    "conf_class",
    "brightness_max_k",
    "daynight",
    "is_static_source",
    "event_class",
]


def _fast_format_utc(s: pd.Series) -> pd.Series:
    return pd.Series(s.to_numpy(dtype="datetime64[m]").astype("U16"), index=s.index) + "Z"


def _fast_format_date(s: pd.Series) -> pd.Series:
    return pd.Series(s.to_numpy(dtype="datetime64[D]").astype("U10"), index=s.index)


def export_table(events: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EXPORT_COLUMNS if c in events.columns]
    out = events[cols].copy()
    for c in ("utc_timestamp", "utc_end"):
        if c in out:
            out[c] = _fast_format_utc(out[c])
    if "acq_date" in out:
        out["acq_date"] = _fast_format_date(out["acq_date"])
    for c in ("latitude", "longitude"):
        out[c] = out[c].round(5)
    for c in ("frp_modis_mw", "frp_viirs_mw", "frp_viirs_calibrated_mw", "calibrated_frp_mw"):
        if c in out:
            out[c] = out[c].round(2)
    return out


# ---------------------------------------------------------------------------
# Sentetik örnek veri üreteci (API anahtarsız demo)
# ---------------------------------------------------------------------------
class SyntheticFIRMSGenerator:
    """
    FIRMS CSV şemasında SENTETİK (simülasyon) MODIS ve VIIRS sıcak nokta verisi üretir.
    Bu veri gerçek uydu gözlemi DEĞİLDİR; yalnızca boru hattını API anahtarı olmadan
    göstermek için kullanılır. Sensör farkları problem tanımındaki niteliksel özelliklere göre
    modellenir: MODIS ~1 km piksel; VIIRS 375 m piksel, bir yangını birden çok alt piksel olarak
    kaydetmesi ve küçük yangınlara daha yüksek duyarlılık. Tüm sayısal simülasyon parametreleri
    (yangın sayısı, tespit eğrileri, geçiş saatleri, konumlar) temsilidir; gerçek istatistik değildir.
    """

    # (ad, çapa noktaları (lat, lon), saçılım derece, rejim ağırlığı, ay ağırlıkları 1..12, medyan güç MW)
    FIRE_REGIMES = [
        ("anız_GD_Anadolu", [(37.85, 41.00), (37.86, 40.66), (36.86, 39.03), (37.19, 40.59), (37.93, 41.70)],
         0.15, 0.45, [0, 0, 0, 0, 1, 6, 9, 6, 2, 1, 0, 0], 4.0),
        ("orman_Akdeniz_Ege", [(36.85, 28.27), (36.79, 31.44), (37.30, 27.80), (36.37, 30.29), (38.25, 27.13), (36.34, 33.40)],
         0.12, 0.30, [0, 0, 1, 1, 2, 4, 8, 9, 6, 2, 1, 0], 15.0),
        ("anız_İç_Anadolu_Trakya", [(39.00, 33.00), (38.60, 35.40), (41.30, 27.00), (39.80, 30.50)],
         0.25, 0.25, [0, 0, 1, 1, 1, 3, 6, 5, 2, 1, 0, 0], 3.0),
    ]
    # Batman çevresinde sentetik statik termal kaynaklar (gerçek tesis konumu iddiası yoktur)
    STATIC_SITES = [(37.80, 41.10), (37.86, 41.30), (37.74, 41.45), (37.68, 41.02), (37.95, 41.55), (37.60, 41.22)]

    # (uydu adı, alet, simülasyonda ilk yıl, ortalama UTC geçiş dakikası) — temsili değerler
    SATELLITES = [
        ("Terra", "MODIS", 2000, 8 * 60 + 20),
        ("Aqua", "MODIS", 2002, 10 * 60 + 50),
        ("Suomi NPP", "VIIRS", 2012, 10 * 60 + 40),
        ("NOAA-20", "VIIRS", 2018, 9 * 60 + 50),
    ]

    def __init__(self, seed: int = 42, fires_per_year: int = 320):
        self.rng = np.random.default_rng(seed)
        self.fires_per_year = fires_per_year

    @staticmethod
    def _p_detect(power: float, center: float, width: float = 0.35) -> float:
        return 1.0 / (1.0 + np.exp(-(np.log10(power) - np.log10(center)) / width))

    def _pixels(self, inst: str, lat: float, lon: float, power: float, ts: datetime, sat: str, daynight: str) -> dict:
        rng = self.rng
        if inst == "MODIS":
            k = 1 + rng.poisson(power / 120.0)
            spread_km, scan = 0.9, rng.uniform(1.0, 2.5)
            frp_scale, noise = 1.0, 0.30
        else:
            k = min(1 + rng.poisson(power / 12.0), 14)
            spread_km, scan = 0.30, rng.uniform(0.39, 0.6)
            frp_scale, noise = 0.85, 0.25
        r = spread_km * np.sqrt(rng.uniform(0, 1, k)) * (1 + 0.15 * np.log1p(k))
        th = rng.uniform(0, 2 * np.pi, k)
        dlat = r * np.cos(th) / KM_PER_DEG_LAT
        dlon = r * np.sin(th) / (KM_PER_DEG_LAT * np.cos(np.radians(lat)))
        share = rng.dirichlet(np.ones(k) * 2.0)
        frp = np.maximum(power * frp_scale * share * rng.lognormal(0, noise, k), 0.3)
        rows = {
            "latitude": np.round(lat + dlat, 5),
            "longitude": np.round(lon + dlon, 5),
            "scan": np.round(np.full(k, scan), 2),
            "track": np.round(np.full(k, scan * 0.8), 2),
            "acq_date": np.full(k, ts.strftime("%Y-%m-%d")),
            "acq_time": np.full(k, ts.hour * 100 + ts.minute),
            "satellite": np.full(k, sat),
            "instrument": np.full(k, inst),
            "version": np.full(k, "SYNTHETIC"),
            "frp": np.round(frp, 2),
            "daynight": np.full(k, daynight),
        }
        if inst == "MODIS":
            rows["brightness"] = np.round(305 + 12 * np.log10(frp + 1) + rng.normal(0, 4, k), 2)
            rows["bright_t31"] = np.round(rows["brightness"] - rng.uniform(10, 25, k), 2)
            rows["confidence"] = np.clip(np.round(45 + 25 * np.log10(frp + 1) + rng.normal(0, 15, k)), 0, 100).astype(int)
        else:
            rows["bright_ti4"] = np.round(np.minimum(315 + 14 * np.log10(frp + 1) + rng.normal(0, 5, k), 367.0), 2)
            rows["bright_ti5"] = np.round(rows["bright_ti4"] - rng.uniform(20, 40, k), 2)
            conf = np.where(frp > 6, "h", "n")
            rows["confidence"] = np.where(rng.uniform(0, 1, k) < 0.08, "l", conf)
        return rows

    def _overpass(self, day: datetime, mean_minute: int, night: bool = False) -> datetime:
        minute = mean_minute + (12 * 60 if night else 0) + int(self.rng.normal(0, 25))
        return day + timedelta(minutes=int(np.clip(minute, 0, 1439)))

    def generate(self, start_year: int = 2000, end_year: int = 2025) -> pd.DataFrame:
        rng = self.rng
        buckets: dict[str, list[dict]] = {"MODIS": [], "VIIRS": []}
        weights = np.array([r[3] for r in self.FIRE_REGIMES], float)
        weights /= weights.sum()
        regime_probs = [np.asarray(r[4], float) / np.sum(r[4]) for r in self.FIRE_REGIMES]

        for year in range(start_year, end_year + 1):
            sats = [s for s in self.SATELLITES if s[2] <= year]
            n_fires = rng.poisson(self.fires_per_year * rng.lognormal(0, 0.25))
            for ri in rng.choice(len(self.FIRE_REGIMES), size=n_fires, p=weights):
                _, anchors, spread, _, _, med_power = self.FIRE_REGIMES[ri]
                month = int(rng.choice(12, p=regime_probs[ri])) + 1
                dim = calendar.monthrange(year, month)[1]
                start = datetime(year, month, int(rng.integers(1, dim + 1)))
                a_lat, a_lon = anchors[rng.integers(len(anchors))]
                lat = a_lat + rng.normal(0, spread)
                lon = a_lon + rng.normal(0, spread)
                power0 = rng.lognormal(np.log(med_power), 1.0)
                duration = int(rng.geometric(0.6 if med_power < 10 else 0.35))
                for d in range(duration):
                    day = start + timedelta(days=d)
                    power = power0 * rng.lognormal(0, 0.4)
                    lat += rng.normal(0, 0.004)
                    lon += rng.normal(0, 0.004)
                    for sat, inst, _, mean_min in sats:
                        center = 9.0 if inst == "MODIS" else 1.5
                        for night in (False, True):
                            p_eff = power * (0.5 if night else 1.0)
                            if rng.uniform() < self._p_detect(p_eff, center):
                                ts = self._overpass(day, mean_min, night)
                                buckets[inst].append(self._pixels(inst, lat, lon, p_eff, ts, sat, "N" if night else "D"))

        # Statik termal kaynaklar (sentetik, yıl boyu sinyal)
        days = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
        years = days.year.to_numpy()
        day_list = days.to_pydatetime()
        for s_lat, s_lon in self.STATIC_SITES:
            base_power = rng.uniform(8, 25)
            for sat, inst, first_year, mean_min in self.SATELLITES:
                p_det = 0.05 if inst == "MODIS" else 0.18
                active = np.nonzero((years >= first_year) & (rng.uniform(0, 1, len(days)) < p_det))[0]
                keep = 1 if inst == "MODIS" else 2
                for di in active:
                    night = bool(rng.uniform() < 0.5)
                    ts = self._overpass(day_list[di], mean_min, night)
                    power = base_power * rng.lognormal(0, 0.3)
                    rows = self._pixels(inst, s_lat + rng.normal(0, 0.0015), s_lon + rng.normal(0, 0.0015),
                                        power, ts, sat, "N" if night else "D")
                    buckets[inst].append({k: v[:keep] for k, v in rows.items()})

        parts = []
        for rows_list in buckets.values():
            if rows_list:
                keys = rows_list[0].keys()
                parts.append(pd.DataFrame({k: np.concatenate([r[k] for r in rows_list]) for k in keys}))
        out = pd.concat(parts, ignore_index=True)
        out["source"] = np.where(out["instrument"] == "MODIS", "SYNTHETIC_MODIS", "SYNTHETIC_VIIRS")
        return out.sort_values(["acq_date", "acq_time"], kind="stable").reset_index(drop=True)