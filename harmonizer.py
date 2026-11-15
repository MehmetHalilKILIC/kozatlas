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
    return np.column_stack([np.cos(phi) * np.cos(lmb), np.cos(phi) * np.sin(lmb), np.sin(phi)])


def _grid_keys(lat: np.ndarray, lon: np.ndarray, cell_km: float, offset: float = 0.0) -> np.ndarray:
    """~cell_km × cell_km grid hücre anahtarı (boylam adımı enleme göre cos düzeltmeli)."""
    dlat = cell_km / KM_PER_DEG_LAT
    iy = np.floor(lat / dlat + offset).astype(np.int64)
    row_center = (iy - offset + 0.5) * dlat
    dlon = cell_km / (KM_PER_DEG_LAT * np.cos(np.radians(row_center)))
    ix = np.floor(lon / dlon + offset).astype(np.int64)
    return iy * 10_000_000 + ix


# ---------------------------------------------------------------------------
# Aşama 1 — Veri çekme ve şema standardizasyonu
# ---------------------------------------------------------------------------
class FIRMSClient:
    """NASA FIRMS Area API istemcisi (CSV)."""

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
        """[start, end] aralığını 10 günlük parçalara bölüp URL listesi döndürür."""
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
        frames, errors = [], []
        for src in sources:
            for url in self.request_plan(src, bbox, start, end):
                try:
                    df = self._get(url)
                except (urllib.error.URLError, RuntimeError, pd.errors.ParserError, TimeoutError) as exc:
                    errors.append(f"{src}: {exc}")
                    continue
                if len(df):
                    df["source"] = src
                    frames.append(df)
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
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    required = {"latitude", "longitude", "acq_date", "acq_time"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Eksik zorunlu FIRMS sütunları: {sorted(missing)}")

    def num(col):
        return pd.to_numeric(d[col], errors="coerce") if col in d else pd.Series(np.nan, index=d.index)

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
    out["utc_timestamp"] = (
        pd.to_datetime(d["acq_date"], errors="coerce")
        + pd.to_timedelta(hhmm // 100, unit="h")
        + pd.to_timedelta(hhmm % 100, unit="m")
    )
    out["acq_date"] = out["utc_timestamp"].dt.normalize()

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

    out = out.dropna(subset=["latitude", "longitude", "utc_timestamp"])
    out = out.sort_values("utc_timestamp", kind="stable").reset_index(drop=True)
    out["point_id"] = np.arange(len(out))
    return out


def filter_aoi(df: pd.DataFrame, bbox) -> pd.DataFrame:
    w, s, e, n = bbox
    m = df["longitude"].between(w, e) & df["latitude"].between(s, n)
    return df[m].reset_index(drop=True)


def filter_dates(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    m = df["acq_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    return df[m].reset_index(drop=True)


def filter_confidence(df: pd.DataFrame, min_level: int) -> pd.DataFrame:
    return df[df["conf_level"] >= int(min_level)].reset_index(drop=True)


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
        ok = (np.abs(t_min[i] - t_min[j]) <= window_min) & (
            haversine_km(lat[i], lon[i], lat[j], lon[j]) <= distance_km
        )
        i, j = i[ok], j[ok]
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
            [",".join(sorted_sats[b] for b in range(len(sorted_sats)) if (m & (1 << b)) and sorted_sats[b]) for m in range(1 << len(sorted_sats))],
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
        stats = sub.groupby(["key", "year"], sort=False).agg(nd=("day", "nunique"), nm=("month", "nunique")).reset_index()
        is_stat_year = (stats["nd"] >= min_days) & (stats["nm"] >= min_months)
        stat_keys = stats.loc[is_stat_year, "key"].unique()
        key_max = stats.groupby("key", sort=False).agg(nd=("nd", "max"), nm=("nm", "max"))

        flag |= np.isin(key, stat_keys)
        idx_key = pd.Index(key)
        days_best = np.maximum(days_best, key_max["nd"].reindex(idx_key, fill_value=0).to_numpy())
        months_best = np.maximum(months_best, key_max["nm"].reindex(idx_key, fill_value=0).to_numpy())

    is_static = pd.Series(flag, index=events.index)
    st = events.assign(persistence_days=days_best, persistence_months=months_best)[flag]
    if st.empty:
        return is_static, pd.DataFrame()
    # Merge adjacent static cells into a single industrial site via micro-cell deduplication
    micro_key = _grid_keys(st["latitude"].to_numpy(float), st["longitude"].to_numpy(float), cell_km * 0.25)
    cells = st.assign(_mk=micro_key).groupby("_mk", sort=False, as_index=False).agg(
        latitude=("latitude", "mean"), longitude=("longitude", "mean")
    )
    cells["utc_timestamp"] = pd.Timestamp("2000-01-01")
    cells["site"] = spatiotemporal_labels(cells, distance_km=cell_km * 1.5, window_min=1.0)
    mk_to_site = dict(zip(cells["_mk"], cells["site"]))
    site_lbl = np.array([mk_to_site[k] for k in micro_key])
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

    id_map = events.set_index("event_idx")
    pts["unified_hotspot_id"] = pts["event_idx"].map(id_map["unified_hotspot_id"])
    pts["is_static_source"] = pts["event_idx"].map(id_map["is_static_source"]).astype(bool)

    n_pts, n_ev = len(pts), len(events)
    dyn = ~events["is_static_source"]
    stats_ = {
        "raw_points": n_pts,
        "raw_modis": int((pts["instrument"] == "MODIS").sum()),
        "raw_viirs": int((pts["instrument"] == "VIIRS").sum()),
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
    df = pd.DataFrame(
        {
            "Raw MODIS Pixels": points[points["instrument"] == "MODIS"].groupby(py)["point_id"].count(),
            "Raw VIIRS Pixels": points[points["instrument"] == "VIIRS"].groupby(py)["point_id"].count(),
            "KOZ Harmonized Events": events.groupby(ey)["unified_hotspot_id"].count(),
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


def export_table(events: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EXPORT_COLUMNS if c in events.columns]
    out = events[cols].copy()
    for c in ("utc_timestamp", "utc_end"):
        if c in out:
            out[c] = out[c].dt.strftime("%Y-%m-%dT%H:%MZ")
    if "acq_date" in out:
        out["acq_date"] = out["acq_date"].dt.strftime("%Y-%m-%d")
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

    def _expand_overpasses(
        self,
        inst: str,
        day_idx: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        power: np.ndarray,
        minute: np.ndarray,
        sat: np.ndarray,
        daynight: np.ndarray,
        keep_max: np.ndarray,
        date_lut: np.ndarray,
    ) -> pd.DataFrame:
        rng = self.rng
        n_ov = len(power)
        if n_ov == 0:
            return pd.DataFrame()
        if inst == "MODIS":
            k_full = 1 + rng.poisson(power / 120.0)
            spread_km, scan_ov = 0.9, rng.uniform(1.0, 2.5, n_ov)
            frp_scale, noise = 1.0, 0.30
        else:
            k_full = np.minimum(1 + rng.poisson(power / 12.0), 14)
            spread_km, scan_ov = 0.30, rng.uniform(0.39, 0.6, n_ov)
            frp_scale, noise = 0.85, 0.25

        starts = np.r_[0, np.cumsum(k_full[:-1])]
        total_px = int(k_full.sum())
        g = rng.gamma(2.0, 1.0, total_px)
        g_sum = np.add.reduceat(g, starts)
        share = g / np.repeat(g_sum, k_full)

        pos = np.arange(total_px) - np.repeat(starts, k_full)
        keep_mask = pos < np.repeat(keep_max, k_full)
        share = share[keep_mask]
        k_kept = np.minimum(k_full, keep_max)
        n_px = int(k_kept.sum())

        k_full_rep = np.repeat(k_full, k_kept)
        lat_rep = np.repeat(lat, k_kept)
        lon_rep = np.repeat(lon, k_kept)
        pow_rep = np.repeat(power, k_kept)
        scan_rep = np.repeat(scan_ov, k_kept)
        min_rep = np.repeat(minute, k_kept)

        r = spread_km * np.sqrt(rng.uniform(0, 1, n_px)) * (1.0 + 0.15 * np.log1p(k_full_rep))
        th = rng.uniform(0, 2 * np.pi, n_px)
        dlat = r * np.cos(th) / KM_PER_DEG_LAT
        dlon = r * np.sin(th) / (KM_PER_DEG_LAT * np.cos(np.radians(lat_rep)))
        frp = np.maximum(pow_rep * frp_scale * share * rng.lognormal(0, noise, n_px), 0.3)

        cols = {
            "latitude": np.round(lat_rep + dlat, 5),
            "longitude": np.round(lon_rep + dlon, 5),
            "scan": np.round(scan_rep, 2),
            "track": np.round(scan_rep * 0.8, 2),
            "acq_date": date_lut[np.repeat(day_idx, k_kept)],
            "acq_time": (min_rep // 60) * 100 + (min_rep % 60),
            "satellite": np.repeat(sat, k_kept),
            "instrument": np.full(n_px, inst),
            "version": np.full(n_px, "SYNTHETIC"),
            "frp": np.round(frp, 2),
            "daynight": np.repeat(daynight, k_kept),
        }
        if inst == "MODIS":
            b1 = np.round(305.0 + 12.0 * np.log10(frp + 1.0) + rng.normal(0, 4, n_px), 2)
            cols["brightness"] = b1
            cols["bright_t31"] = np.round(b1 - rng.uniform(10, 25, n_px), 2)
            cols["confidence"] = np.clip(
                np.round(45.0 + 25.0 * np.log10(frp + 1.0) + rng.normal(0, 15, n_px)), 0, 100
            ).astype(int)
        else:
            b1 = np.round(np.minimum(315.0 + 14.0 * np.log10(frp + 1.0) + rng.normal(0, 5, n_px), 367.0), 2)
            cols["bright_ti4"] = b1
            cols["bright_ti5"] = np.round(b1 - rng.uniform(20, 40, n_px), 2)
            conf = np.where(frp > 6.0, "h", "n")
            cols["confidence"] = np.where(rng.uniform(0, 1, n_px) < 0.08, "l", conf)
        return pd.DataFrame(cols)

    def generate(self, start_year: int = 2000, end_year: int = 2025) -> pd.DataFrame:
        rng = self.rng
        epoch = np.datetime64(f"{start_year:04d}-01-01", "D")
        date_lut = (
            pd.date_range(f"{start_year}-01-01", f"{end_year + 1}-03-01", freq="D")
            .strftime("%Y-%m-%d")
            .to_numpy()
        )
        weights = np.array([r[3] for r in self.FIRE_REGIMES], float)
        weights /= weights.sum()

        yr_list, ri_list = [], []
        for year in range(start_year, end_year + 1):
            n_fires = int(rng.poisson(self.fires_per_year * rng.lognormal(0, 0.25)))
            yr_list.append(np.full(n_fires, year, dtype=np.int32))
            ri_list.append(rng.choice(len(self.FIRE_REGIMES), size=n_fires, p=weights))
        fire_years = np.concatenate(yr_list)
        fire_ris = np.concatenate(ri_list)

        f_day0, f_yr, f_lat0, f_lon0, f_pow0, f_dur = [], [], [], [], [], []
        for ri, (_, anchors, spread, _, mw, med_power) in enumerate(self.FIRE_REGIMES):
            m = fire_ris == ri
            cnt = int(m.sum())
            if cnt == 0:
                continue
            yrs = fire_years[m]
            mw_p = np.asarray(mw, float)
            mw_p /= mw_p.sum()
            months = rng.choice(12, size=cnt, p=mw_p) + 1
            ym_str = np.char.add(np.char.add(yrs.astype("U4"), "-"), np.char.zfill(months.astype("U2"), 2))
            m_start = ym_str.astype("datetime64[M]").astype("datetime64[D]")
            m_next = (ym_str.astype("datetime64[M]") + np.timedelta64(1, "M")).astype("datetime64[D]")
            dim = (m_next - m_start).astype(np.int32)
            day0 = (m_start - epoch).astype(np.int32) + rng.integers(0, dim)

            anc = np.asarray(anchors, float)
            ai = rng.integers(len(anc), size=cnt)
            f_day0.append(day0)
            f_yr.append(yrs)
            f_lat0.append(anc[ai, 0] + rng.normal(0, spread, cnt))
            f_lon0.append(anc[ai, 1] + rng.normal(0, spread, cnt))
            f_pow0.append(rng.lognormal(np.log(med_power), 1.0, cnt))
            f_dur.append(rng.geometric(0.6 if med_power < 10 else 0.35, cnt))

        day0_all = np.concatenate(f_day0)
        yr_all = np.concatenate(f_yr)
        lat0_all = np.concatenate(f_lat0)
        lon0_all = np.concatenate(f_lon0)
        pow0_all = np.concatenate(f_pow0)
        dur_all = np.concatenate(f_dur)

        f_starts = np.r_[0, np.cumsum(dur_all[:-1])]
        total_fd = int(dur_all.sum())
        d_off = np.arange(total_fd, dtype=np.int32) - np.repeat(f_starts, dur_all)
        fd_day = np.repeat(day0_all, dur_all) + d_off
        fd_year = np.repeat(yr_all, dur_all)
        fd_power = np.repeat(pow0_all, dur_all) * rng.lognormal(0, 0.4, total_fd)

        s_lat = rng.normal(0, 0.004, total_fd)
        s_lon = rng.normal(0, 0.004, total_fd)
        cs_lat, cs_lon = np.cumsum(s_lat), np.cumsum(s_lon)
        fd_lat = np.repeat(lat0_all, dur_all) + cs_lat - np.repeat(cs_lat[f_starts] - s_lat[f_starts], dur_all)
        fd_lon = np.repeat(lon0_all, dur_all) + cs_lon - np.repeat(cs_lon[f_starts] - s_lon[f_starts], dur_all)

        ov_buckets: dict[str, list[tuple]] = {"MODIS": [], "VIIRS": []}
        for sat, inst, first_year, mean_min in self.SATELLITES:
            center = 9.0 if inst == "MODIS" else 1.5
            ymask = fd_year >= first_year
            if not ymask.any():
                continue
            d_sub, lat_sub, lon_sub, pow_sub = fd_day[ymask], fd_lat[ymask], fd_lon[ymask], fd_power[ymask]
            for night in (False, True):
                p_eff = pow_sub * (0.5 if night else 1.0)
                det = rng.uniform(0, 1, len(p_eff)) < self._p_detect(p_eff, center)
                n_det = int(det.sum())
                if n_det == 0:
                    continue
                mins = np.clip(
                    mean_min + (720 if night else 0) + rng.normal(0, 25, n_det).astype(np.int32), 0, 1439
                )
                ov_buckets[inst].append((
                    d_sub[det],
                    lat_sub[det],
                    lon_sub[det],
                    p_eff[det],
                    mins,
                    np.full(n_det, sat),
                    np.full(n_det, "N" if night else "D"),
                    np.full(n_det, 99, dtype=np.int32),
                ))

        # Static thermal sources (synthetic year-round signal)
        days = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
        years = days.year.to_numpy()
        n_days = len(days)
        for s_lat_c, s_lon_c in self.STATIC_SITES:
            base_power = rng.uniform(8, 25)
            for sat, inst, first_year, mean_min in self.SATELLITES:
                p_det = 0.05 if inst == "MODIS" else 0.18
                active = np.nonzero((years >= first_year) & (rng.uniform(0, 1, n_days) < p_det))[0].astype(np.int32)
                n_act = len(active)
                if n_act == 0:
                    continue
                night_arr = rng.uniform(0, 1, n_act) < 0.5
                mins = np.clip(
                    mean_min + np.where(night_arr, 720, 0) + rng.normal(0, 25, n_act).astype(np.int32), 0, 1439
                )
                ov_buckets[inst].append((
                    active,
                    s_lat_c + rng.normal(0, 0.0015, n_act),
                    s_lon_c + rng.normal(0, 0.0015, n_act),
                    base_power * rng.lognormal(0, 0.3, n_act),
                    mins,
                    np.full(n_act, sat),
                    np.where(night_arr, "N", "D"),
                    np.full(n_act, 1 if inst == "MODIS" else 2, dtype=np.int32),
                ))

        parts = []
        for inst, chunks in ov_buckets.items():
            if not chunks:
                continue
            cols_cat = [np.concatenate([c[i] for c in chunks]) for i in range(8)]
            parts.append(self._expand_overpasses(inst, *cols_cat, date_lut))
        out = pd.concat(parts, ignore_index=True)
        out["source"] = np.where(out["instrument"] == "MODIS", "SYNTHETIC_MODIS", "SYNTHETIC_VIIRS")
        return out.sort_values(["acq_date", "acq_time"], kind="stable").reset_index(drop=True)
