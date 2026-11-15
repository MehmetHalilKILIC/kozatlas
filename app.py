"""
KOZ Atlas — MODIS & VIIRS Thermal Anomaly and Active Fire Harmonization Platform
2026 NASA Space Apps Challenge · Harmonization of MODIS and VIIRS Hot Spots · Team: Kirvecanlar

Run command:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
from datetime import date, timedelta
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Fullscreen, MiniMap
from streamlit_folium import st_folium

import harmonizer as hz

# ---------------------------------------------------------------------------
# Constants & Theme
# ---------------------------------------------------------------------------
APP_URL = "https://koz-atlas-gbqjscepvmqcaygzbjlxhq.streamlit.app/"
REPO_URL = "https://github.com/MehmetHalilKILIC/koz-atlas"
CHALLENGE_URL = "https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/"
FIRMS_GUIDE_URL = "https://github.com/nasa/spaceapps/discussions/2016"
FIRMS_KEY_URL = "https://firms.modaps.eosdis.nasa.gov/api/map_key/"

BG, CARD, EMBER, AMBER = "#0B0F19", "#1E293B", "#FF5722", "#FF9800"
C_MODIS, C_VIIRS, C_KOZ, C_STATIC = "#38BDF8", "#FACC15", "#FF5722", "#A855F7"
MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CONF_OPTIONS = {"low (all)": 0, "nominal": 1, "high": 2}
MAP_LAYER_CAP = 800
CUSTOM_AOI = "Custom Bounding Box"
DARK_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
SAT_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

st.set_page_config(page_title="KOZ Atlas — MODIS & VIIRS Fire Harmonization", page_icon="🔥", layout="wide")

st.markdown(
    f"""
<style>
.stApp {{ background: radial-gradient(ellipse at top, #131a2e 0%, {BG} 55%); }}
section[data-testid="stSidebar"] {{ background: #0E1424; border-right: 1px solid #1f2a44; }}
.block-container {{ padding-top: 1.6rem; }}
.koz-title {{ font-size: 2.3rem; font-weight: 800; letter-spacing: .5px; margin: 0;
    background: linear-gradient(90deg, {EMBER}, {AMBER}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.koz-sub {{ color: #94A3B8; font-size: .95rem; margin-top: .1rem; }}
.koz-sub a {{ color: {AMBER}; text-decoration: none; }}
.kpi {{ background: rgba(30,41,59,.72); border: 1px solid rgba(255,87,34,.25); border-radius: 14px;
    padding: 14px 16px; height: 100%; backdrop-filter: blur(6px); }}
.kpi .lbl {{ color: #94A3B8; font-size: .78rem; text-transform: uppercase; letter-spacing: .6px; }}
.kpi .val {{ color: #F8FAFC; font-size: 1.75rem; font-weight: 700; line-height: 1.25; }}
.kpi .det {{ color: {AMBER}; font-size: .8rem; }}
.note {{ background: rgba(30,41,59,.6); border-left: 3px solid {EMBER}; border-radius: 6px; padding: 10px 14px;
    color: #CBD5E1; font-size: .88rem; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
.stTabs [aria-selected="true"] {{ color: {EMBER} !important; }}
</style>
""",
    unsafe_allow_html=True,
)


def kpi(col, label: str, value: str, detail: str = "") -> None:
    col.markdown(
        f'<div class="kpi"><div class="lbl">{label}</div><div class="val">{value}</div><div class="det">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,41,59,.35)",
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
        font=dict(color="#E2E8F0"),
    )
    return fig


def fmt_int(x) -> str:
    return f"{int(round(x)):,}"


# ---------------------------------------------------------------------------
# Data Loading (Cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading demo archive…")
def load_synthetic(y0: int, y1: int, seed: int = 42) -> pd.DataFrame:
    p = Path(__file__).resolve().parent / "data" / "demo_firms.parquet"
    if p.exists() and seed == 42:
        df = pd.read_parquet(p)
        return df[(df["acq_date"].dt.year >= y0) & (df["acq_date"].dt.year <= y1)].reset_index(drop=True)
    return hz.standardize_firms(hz.SyntheticFIRMSGenerator(seed=seed).generate(y0, y1))


@st.cache_data(show_spinner="Fetching telemetry from NASA FIRMS Area API…", ttl=3600)
def load_firms(map_key: str, sources: tuple, bbox: tuple, d0: date, d1: date) -> tuple[pd.DataFrame, list]:
    raw, errors = hz.FIRMSClient(map_key).fetch(list(sources), bbox, d0, d1)
    return hz.standardize_firms(raw), errors


@st.cache_data(show_spinner="Reading CSV files…")
def load_csv(blobs: tuple) -> pd.DataFrame:
    frames = [pd.read_csv(io.BytesIO(b)) for b in blobs]
    return hz.standardize_firms(pd.concat(frames, ignore_index=True))


@st.cache_data(show_spinner="Executing Spatiotemporal Harmonization Pipeline (Stages 1–4)…")
def run_cached(std: pd.DataFrame, bbox: tuple, d0: date, d1: date, params: dict) -> hz.PipelineResult:
    data = hz.filter_dates(hz.filter_aoi(std, bbox), d0, d1) if not std.empty else std
    return hz.run_pipeline(data, hz.PipelineParams(**params))


# ---------------------------------------------------------------------------
# Sidebar: Control Panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 🔥 KOZ Atlas Control Panel")

    st.markdown("**1 · Area of Interest (AOI)**")
    aoi_choices = list(hz.AOI_PRESETS)[:3] + [CUSTOM_AOI]
    aoi_name = st.selectbox("Region", aoi_choices, index=0, label_visibility="collapsed")
    if aoi_name == CUSTOM_AOI:
        c1, c2 = st.columns(2)
        west = c1.number_input("West (lon)", value=40.0, format="%.3f")
        east = c2.number_input("East (lon)", value=42.5, format="%.3f")
        south = c1.number_input("South (lat)", value=37.0, format="%.3f")
        north = c2.number_input("North (lat)", value=38.5, format="%.3f")
        bbox = (min(west, east), min(south, north), max(west, east), max(south, north))
    else:
        bbox = hz.AOI_PRESETS[aoi_name]
    st.caption(f"Bounding Box (W,S,E,N): `{hz.FIRMSClient.area_string(bbox)}`")

    st.markdown("**2 · Data Source**")
    mode = st.radio(
        "Data source",
        ["Built-in synthetic demo (keyless)", "Live NASA FIRMS Area API", "Upload FIRMS CSV files"],
        label_visibility="collapsed",
    )

    std = pd.DataFrame()
    fetch_errors: list[str] = []
    if mode.startswith("Built-in"):
        y0, y1 = st.slider("Simulation years", 2000, 2025, (2000, 2025))
        std = load_synthetic(y0, y1)
    elif mode.startswith("Live"):
        map_key = st.text_input("FIRMS MAP_KEY", type="password", help="Get free key: " + FIRMS_KEY_URL)
        sources = st.multiselect("FIRMS sources", hz.FIRMS_SOURCES, default=["MODIS_NRT", "VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT"])
        today = date.today()
        rng_api = st.date_input("API date range", (today - timedelta(days=9), today), max_value=today)
        d_api = rng_api if isinstance(rng_api, tuple) and len(rng_api) == 2 else (rng_api, rng_api)
        n_req = sum(len(hz.FIRMSClient("x").request_plan(s, bbox, d_api[0], d_api[1])) for s in sources)
        st.caption(f"Plan: {n_req} requests (NASA Area API accepts maximum 5-day slices). NRT sources cover recent observations.")
        if st.button("Fetch Telemetry", type="primary", disabled=not (map_key and sources)):
            st.session_state["api_args"] = (map_key, tuple(sources), bbox, d_api[0], d_api[1])
        if "api_args" in st.session_state:
            std, fetch_errors = load_firms(*st.session_state["api_args"])
    else:
        files = st.file_uploader("FIRMS CSV (MODIS and/or VIIRS)", type=["csv"], accept_multiple_files=True)
        if files:
            std = load_csv(tuple(f.getvalue() for f in files))

    if not std.empty:
        dmin, dmax = std["acq_date"].min().date(), std["acq_date"].max().date()
        dr = st.date_input("Analysis date range", (dmin, dmax), min_value=dmin, max_value=dmax)
        d0, d1 = dr if isinstance(dr, tuple) and len(dr) == 2 else (dmin, dmax)
    else:
        d0 = d1 = date.today()

    st.markdown("**3 · Harmonization Parameters**")
    distance_km = st.slider("Spatial threshold Δd (km)", 0.5, 2.5, 1.2, 0.1, help="MODIS ~1 km ground footprint")
    window_min = st.slider("Temporal window ±Δt (minutes)", 10, 180, 60, 5)
    conf_label = st.select_slider("Minimum confidence threshold", list(CONF_OPTIONS), value="nominal")
    hide_static = st.toggle("Filter static industrial thermal sources (flares)", value=True)

    with st.expander("Advanced: Calibration & Static Sources"):
        cal_method = st.selectbox(
            "FRP calibration model",
            ["loglog", "linear", "identity"],
            format_func={"loglog": "Power-law (log-log OLS)", "linear": "Linear (through origin)", "identity": "No calibration (Identity)"}.get,
        )
        static_days = st.number_input("Static: min distinct days/year", 2, 60, 5)
        static_months = st.number_input("Static: min distinct months/year", 1, 12, 4)
        static_cell = st.slider("Static grid cell (km)", 0.5, 2.0, 1.0, 0.1)

    params = dict(
        distance_km=float(distance_km),
        window_min=float(window_min),
        min_conf_level=CONF_OPTIONS[conf_label],
        calibration_method=cal_method,
        static_cell_km=float(static_cell),
        static_min_days=int(static_days),
        static_min_months=int(static_months),
    )

# ---------------------------------------------------------------------------
# Header & Project Identity
# ---------------------------------------------------------------------------
st.markdown('<p class="koz-title">KOZ Atlas</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="koz-sub">Multi-Satellite Active Fire &amp; Thermal Anomaly Harmonization Platform · '
    f'<a href="{CHALLENGE_URL}" target="_blank">2026 NASA Space Apps Challenge — Harmonization of MODIS and VIIRS Hot Spots</a> · '
    f'Team <b>Kirvecanlar</b> · <a href="{REPO_URL}" target="_blank">GitHub</a></p>',
    unsafe_allow_html=True,
)

if mode.startswith("Built-in"):
    st.warning(
        "Demo Mode: Displayed data represents a synthetic simulation under the FIRMS CSV schema, not live satellite telemetry. "
        "For scientific analysis, provide a FIRMS MAP_KEY in the sidebar or upload FIRMS archive CSV files.",
        icon="⚠️",
    )
for err in fetch_errors[:5]:
    st.error(err)

if std.empty:
    st.info("Please select a data source: live FIRMS API (requires MAP_KEY) or upload FIRMS CSV files.")
    st.stop()

res = run_cached(std, tuple(bbox), d0, d1, params)
if res.events.empty:
    st.info("No thermal anomalies detected within the selected region / date range / confidence threshold.")
    st.stop()

events_all, points_all = res.events, res.points
events_view = events_all[~events_all["is_static_source"]] if hide_static else events_all
points_view = points_all[~points_all["is_static_source"]] if hide_static else points_all

# ---------------------------------------------------------------------------
# Top KPI Metric Cards
# ---------------------------------------------------------------------------
s = res.stats
k1, k2, k3, k4, k5 = st.columns(5)
kpi(k1, "Raw Hotspots (MODIS+VIIRS)", fmt_int(s["raw_points"]), f"MODIS {fmt_int(s['raw_modis'])} · VIIRS {fmt_int(s['raw_viirs'])}")
kpi(k2, "KOZ Harmonized Events", fmt_int(s["events"]), f"Dynamic {fmt_int(s['events'] - s['static_events'])}")
kpi(k3, "Duplication Reduction", f"{s['dedup_ratio'] * 100:.1f}%", f"{fmt_int(s['raw_points'] - s['events'])} duplicate pixels merged")
kpi(k4, "Static Industrial Sources", fmt_int(s["static_sites"]), f"{fmt_int(s['static_events'])} events isolated")
kpi(k5, "Mean Calibrated FRP (Dynamic)", f"{s['mean_calibrated_frp']:.1f} MW", res.calibration.method)
st.write("")

tab_map, tab_cal, tab_proof, tab_dl = st.tabs(
    ["🗺️ Interactive GIS Satellite Map", "📅 Burning Activity Calendar", "🔬 Sensor Harmonization Evidence", "⬇️ Data Download Center"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Interactive Satellite Map
# ---------------------------------------------------------------------------
with tab_map:
    months = events_view["acq_date"].dt.to_period("M")
    month_counts = months.value_counts().sort_index()
    if month_counts.empty:
        st.info("No events to display on the map.")
    else:
        options = [str(p) for p in month_counts.index]
        last_year = month_counts.index.max().year
        recent = month_counts[[p.year == last_year for p in month_counts.index]]
        default = str(recent.idxmax())
        c1, c2 = st.columns([1, 3])
        sel_month = c1.selectbox("Observation period (month)", options[::-1], index=options[::-1].index(default))
        basemap = c1.radio("Basemap", ["Dark Canvas (Esri Dark Gray)", "Satellite Imagery (Esri World Imagery)"])
        c1.markdown(
            f"""<div class="note">
            <b style="color:{C_MODIS}">●</b> Raw MODIS (1 km circle)<br>
            <b style="color:{C_VIIRS}">●</b> Raw VIIRS (375 m circle)<br>
            <b style="color:{C_KOZ}">●</b> KOZ harmonized event (size ∝ FRP)<br>
            <b style="color:{C_STATIC}">★</b> Static industrial emitter (entire observation period)<br>
            Toggle layers using the top-right layer control.</div>""",
            unsafe_allow_html=True,
        )

        per = pd.Period(sel_month)
        ev_m = events_view[events_view["acq_date"].dt.to_period("M") == per]
        pt_m = points_view[points_view["acq_date"].dt.to_period("M") == per]
        if len(ev_m) > MAP_LAYER_CAP:
            ev_m = ev_m.nlargest(MAP_LAYER_CAP, "calibrated_frp_mw")
            pt_m = pt_m[pt_m["unified_hotspot_id"].isin(ev_m["unified_hotspot_id"])]
            c1.caption(f"Showing top {MAP_LAYER_CAP} events by FRP for performance.")

        center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]
        zoom = 6 if (bbox[2] - bbox[0]) > 8 else 8
        tile_url, tile_attr = (DARK_TILES, "Esri, HERE, Garmin, OpenStreetMap contributors") if basemap.startswith("Dark") \
            else (SAT_TILES, "Esri World Imagery")
        fmap = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
        folium.TileLayer(tiles=tile_url, attr=tile_attr, name=basemap, max_zoom=16 if basemap.startswith("Dark") else 19).add_to(fmap)
        folium.Rectangle([[bbox[1], bbox[0]], [bbox[3], bbox[2]]], color=EMBER, weight=1, fill=False, dash_array="6").add_to(fmap)

        lyr_m = folium.FeatureGroup(name="Raw MODIS (1 km)", show=True)
        lyr_v = folium.FeatureGroup(name="Raw VIIRS (375 m)", show=True)
        lyr_k = folium.FeatureGroup(name="KOZ Harmonized Events", show=True)
        lyr_s = folium.FeatureGroup(name="Static Industrial Sources", show=True)

        for r in pt_m.itertuples(index=False):
            is_m = r.instrument == "MODIS"
            folium.Circle(
                [r.latitude, r.longitude],
                radius=500 if is_m else 187.5,
                color=C_MODIS if is_m else C_VIIRS,
                weight=1,
                fill=True,
                fill_opacity=0.15 if is_m else 0.45,
                tooltip=f"{r.instrument} · {r.satellite} · {r.utc_timestamp:%Y-%m-%d %H:%M}Z · FRP {r.frp:.1f} MW",
            ).add_to(lyr_m if is_m else lyr_v)

        frp_ref = max(float(ev_m["calibrated_frp_mw"].quantile(0.95)) if len(ev_m) else 1.0, 1.0)
        for r in ev_m.itertuples(index=False):
            popup = f"""
            <div style="font-family:sans-serif;font-size:12px;min-width:240px">
            <b style="color:{EMBER}">{r.unified_hotspot_id}</b><br>
            <b>Class:</b> {r.event_class}<br>
            <b>Sensor:</b> {r.sensor_label}<br>
            <b>Merged Pixels:</b> {r.merged_pixel_count} (MODIS {r.n_modis}, VIIRS {r.n_viirs})<br>
            <b>Satellites:</b> {r.satellites}<br>
            <b>Time (UTC):</b> {r.utc_timestamp:%Y-%m-%d %H:%M} – {r.utc_end:%H:%M}<br>
            <b>FRP MODIS / VIIRS:</b> {r.frp_modis_mw:.1f} / {r.frp_viirs_mw:.1f} MW<br>
            <b>Calibrated FRP:</b> {r.calibrated_frp_mw:.1f} MW<br>
            <b>MODIS-Equivalent Weight:</b> {r.modis_eq_weight:.2f}<br>
            <b>Confidence:</b> {r.conf_class}<br>
            <b>Coordinates:</b> {r.latitude:.4f}, {r.longitude:.4f}
            </div>"""
            folium.CircleMarker(
                [r.latitude, r.longitude],
                radius=float(4 + 10 * min(r.calibrated_frp_mw / frp_ref, 1.0)),
                color=C_STATIC if r.is_static_source else C_KOZ,
                weight=1.5,
                fill=True,
                fill_color=AMBER if not r.is_static_source else C_STATIC,
                fill_opacity=0.75,
                popup=folium.Popup(popup, max_width=320),
                tooltip=f"{r.unified_hotspot_id} · {r.calibrated_frp_mw:.1f} MW",
            ).add_to(lyr_k)

        for r in res.static_sites.itertuples(index=False):
            folium.Marker(
                [r.latitude, r.longitude],
                icon=folium.Icon(color="purple", icon="industry", prefix="fa"),
                popup=folium.Popup(
                    f"<b>{r.static_site_id}</b> — Static Industrial Thermal Source<br>"
                    f"Events: {r.event_count} · Active Days: {r.active_days}<br>"
                    f"Peak Annual Activity: {r.max_days_in_year} days / {r.max_months_in_year} months<br>"
                    f"First–Last Seen: {r.first_seen:%Y-%m-%d} – {r.last_seen:%Y-%m-%d}<br>"
                    f"Mean Calibrated FRP: {r.mean_calibrated_frp_mw:.1f} MW",
                    max_width=300,
                ),
                tooltip=r.static_site_id,
            ).add_to(lyr_s)

        for lyr in (lyr_m, lyr_v, lyr_k, lyr_s):
            lyr.add_to(fmap)
        folium.LayerControl(collapsed=False).add_to(fmap)
        Fullscreen().add_to(fmap)
        MiniMap(toggle_display=True, tile_layer=folium.TileLayer(tiles=DARK_TILES, attr="Esri")).add_to(fmap)
        with c2:
            st_folium(fmap, height=640, use_container_width=True, returned_objects=[], key=f"map_{sel_month}_{basemap}")
            st.caption(
                f"{sel_month}: {fmt_int(len(pt_m))} raw pixels → {fmt_int(len(ev_m))} KOZ events · "
                f"Δd ≤ {distance_km:.1f} km, |Δt| ≤ {window_min} min"
            )

# ---------------------------------------------------------------------------
# Tab 2 — Burning Activity Calendar
# ---------------------------------------------------------------------------
with tab_cal:
    c1, c2, c3 = st.columns([2, 1, 1])
    metric = c1.selectbox("Calendar metric", list(hz.CALENDAR_METRICS), format_func=hz.CALENDAR_METRICS.get)
    mat = hz.monthly_matrix(events_view, metric, cell_km=params["static_cell_km"])
    years = list(mat.index)
    sel_year = c2.selectbox("Year", years[::-1])
    sel_mon = c3.selectbox("Month", list(range(1, 13)), index=int(events_view["acq_date"].dt.month.mode().iloc[0]) - 1,
                           format_func=lambda m: MONTHS_EN[m - 1])

    st.markdown(
        f"""<div class="note"><b>Metric Definition.</b> <i>MODIS-equivalent active fire cell-days</i>: events are aggregated
        at ~{params['static_cell_km']:.1f} km grid cell × UTC day level (recurrent orbital passes within the same day are deduplicated);
        each cell-day is weighted by the empirical MODIS detection probability (1 if observed by MODIS; data-driven probability based on calibrated FRP if VIIRS-only).
        This enables unbiased comparison between 2000–2012 (MODIS-only) and post-2012 (MODIS+VIIRS).
        {' Excluding static sources.' if hide_static else ' Including static sources.'}</div>""",
        unsafe_allow_html=True,
    )

    zfmt = ".0f" if metric != "frp" else ".3s"
    fig = px.imshow(
        mat.values,
        x=MONTHS_EN,
        y=[str(y) for y in mat.index],
        color_continuous_scale=[[0, "#0B0F19"], [0.15, "#3b1408"], [0.45, "#b8350f"], [0.75, EMBER], [1, "#FFD08A"]],
        aspect="auto",
        labels=dict(color=hz.CALENDAR_METRICS[metric]),
        text_auto=zfmt if len(mat) <= 30 else False,
    )
    fig.update_layout(title=f"Burning Activity Calendar — {hz.CALENDAR_METRICS[metric]} (Year × Month)")
    fig.add_shape(type="rect", x0=sel_mon - 1.5, x1=sel_mon - 0.5, y0=years.index(sel_year) - 0.5,
                  y1=years.index(sel_year) + 0.5, line=dict(color="#FFFFFF", width=2))
    st.plotly_chart(style_fig(fig, height=max(380, 24 * len(mat) + 120)), width="stretch")

    an = hz.month_anomaly(mat, sel_year, sel_mon)
    a1, a2, a3, a4 = st.columns(4)
    val_fmt = (lambda v: f"{v:,.0f}") if metric != "frp" else (lambda v: f"{v:,.0f} MW")
    kpi(a1, f"{MONTHS_EN[sel_mon - 1]} {sel_year}", val_fmt(an["current"]), hz.CALENDAR_METRICS[metric])
    kpi(a2, "Historical Median", val_fmt(an["median"]) if an["n_years"] else "—", f"{an['n_years']} years (excluding selected year)")
    ratio_txt = f"{an['ratio']:.2f}×" if np.isfinite(an.get("ratio", np.nan)) else "—"
    kpi(a3, "Ratio to Median", ratio_txt, "1.00× = historical baseline")
    pct = an.get("percentile", np.nan)
    status = "—" if not np.isfinite(pct) else ("Above normal" if an["current"] > an["q75"] else "Below normal" if an["current"] < an["q25"] else "Normal range (Q1–Q3)")
    kpi(a4, "Percentile / Status", f"{pct:.0f}%" if np.isfinite(pct) else "—", status)
    st.write("")

    b1, b2 = st.columns(2)
    hist = mat[sel_mon]
    colors = [EMBER if y == sel_year else "#475569" for y in hist.index]
    fig_b = go.Figure(go.Bar(x=[str(y) for y in hist.index], y=hist.values, marker_color=colors, name=MONTHS_EN[sel_mon - 1]))
    if an["n_years"]:
        fig_b.add_hline(y=an["median"], line=dict(color=AMBER, dash="dash"), annotation_text="historical median")
        fig_b.add_hrect(y0=an["q25"], y1=an["q75"], fillcolor=AMBER, opacity=0.08, line_width=0)
    fig_b.update_layout(title=f"{MONTHS_EN[sel_mon - 1]} — Multi-Year Historical Comparison")
    b1.plotly_chart(style_fig(fig_b, 380), width="stretch")

    clim = mat.drop(index=sel_year, errors="ignore")
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=MONTHS_EN, y=clim.quantile(0.75).values, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_c.add_trace(go.Scatter(x=MONTHS_EN, y=clim.quantile(0.25).values, fill="tonexty", fillcolor="rgba(255,152,0,.15)",
                               line=dict(width=0), name="Q1–Q3 Envelope"))
    fig_c.add_trace(go.Scatter(x=MONTHS_EN, y=clim.median().values, line=dict(color=AMBER, dash="dash"), name="Historical Median"))
    fig_c.add_trace(go.Scatter(x=MONTHS_EN, y=mat.loc[sel_year].values, line=dict(color=EMBER, width=3), mode="lines+markers", name=str(sel_year)))
    fig_c.update_layout(title=f"Seasonal Profile — {sel_year} vs Historical Distribution")
    b2.plotly_chart(style_fig(fig_c, 380), width="stretch")

    daily = events_view[events_view["acq_date"].dt.year == sel_year].groupby("acq_date").agg(
        events=("unified_hotspot_id", "size"), frp=("calibrated_frp_mw", "sum")
    )
    if not daily.empty:
        fig_d = go.Figure(go.Bar(x=daily.index, y=daily["events"], marker_color=EMBER, name="Daily KOZ Events"))
        fig_d.update_layout(title=f"{sel_year} — Daily Harmonized Event Count")
        st.plotly_chart(style_fig(fig_d, 300), width="stretch")

# ---------------------------------------------------------------------------
# Tab 3 — Sensor Harmonization Evidence
# ---------------------------------------------------------------------------
with tab_proof:
    ann = hz.annual_series(points_view, events_view, cell_km=params["static_cell_km"])
    series_cols = {
        "Raw Total (MODIS+VIIRS)": "#64748B",
        "Raw MODIS Pixels": C_MODIS,
        "KOZ Harmonized Events": AMBER,
        "KOZ MODIS-Equivalent Cell-Days": EMBER,
    }
    fig_a = go.Figure()
    for col, color in series_cols.items():
        fig_a.add_trace(go.Scatter(x=ann.index, y=ann[col], name=col, mode="lines+markers",
                                   line=dict(color=color, width=3 if "MODIS-Equivalent" in col else 2)))
    fig_a.add_bar(x=ann.index, y=ann["Raw VIIRS Pixels"], name="Raw VIIRS Pixels", marker_color=C_VIIRS, opacity=0.25)
    if ann.index.min() < 2012 <= ann.index.max():
        fig_a.add_vline(x=2011.5, line=dict(color="#F8FAFC", dash="dot"), annotation_text="VIIRS Data Stream Begins (2012)")
    fig_a.update_layout(title="Annual Time Series Before and After Harmonization", xaxis_title="Year", yaxis_title="Count")
    st.plotly_chart(style_fig(fig_a, 460), width="stretch")

    if ann.index.min() < 2012 <= ann.index.max():
        br = pd.DataFrame(
            {
                "Series": list(series_cols),
                "2012+ Mean / Pre-2012 Mean": [hz.break_ratio(ann[c]) for c in series_cols],
            }
        )
        c1, c2 = st.columns([1, 1.4])
        c1.markdown("**Sensor Break Ratio** (1.00 = No Artificial Break)")
        c1.dataframe(br.style.format({"2012+ Mean / Pre-2012 Mean": "{:.2f}"}), hide_index=True, width="stretch")
        c2.markdown(
            f"""<div class="note">The raw total hotspot count spikes dramatically in 2012 due to the introduction of VIIRS.
            The KOZ pipeline resolves this by (i) clustering 375 m VIIRS sub-pixels and coincident 1 km MODIS into a single event footprint,
            (ii) deduplicating multiple daily satellite passes to a cell-day resolution, and
            (iii) weighting sub-pixel VIIRS-only detections by their empirical MODIS detectability probability.
            The MODIS-only historical series serves as the baseline ground truth reference.</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.caption("The analysis date range must span both pre-2012 and post-2012 to evaluate the sensor break ratio.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    both = events_all[(events_all["frp_modis_mw"] > 0) & (events_all["frp_viirs_mw"] > 0)]
    with c1:
        if len(both):
            sample = both.sample(min(len(both), 4000), random_state=0)
            fig_s = go.Figure(go.Scattergl(
                x=sample["frp_viirs_mw"], y=sample["frp_modis_mw"], mode="markers",
                marker=dict(size=5, color=EMBER, opacity=0.35), name="BOTH Events",
            ))
            xs = np.logspace(np.log10(max(both["frp_viirs_mw"].min(), 0.1)), np.log10(both["frp_viirs_mw"].max()), 60)
            fig_s.add_trace(go.Scatter(x=xs, y=res.calibration.apply(xs), line=dict(color=AMBER, width=3), name="Fitted Calibration"))
            fig_s.add_trace(go.Scatter(x=xs, y=xs, line=dict(color="#94A3B8", dash="dot"), name="1:1 Identity"))
            fig_s.update_xaxes(type="log", title="VIIRS Cluster FRP (MW, intra-pass sum)")
            fig_s.update_yaxes(type="log", title="MODIS FRP (MW)")
            fig_s.update_layout(title="MODIS vs VIIRS FRP Correlation (Coincident Events)")
            st.plotly_chart(style_fig(fig_s, 420), width="stretch")
        st.code(res.calibration.describe(), language=None)
    with c2:
        curve = res.detection_curve
        if curve.available:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=10 ** curve.log_frp_centers, y=curve.raw_ratio, mode="markers",
                                       marker=dict(color=C_MODIS, size=9), name="P(Observed by MODIS | VIIRS)"))
            fig_p.add_trace(go.Scatter(x=10 ** curve.log_frp_centers, y=curve.p_detect, line=dict(color=EMBER, width=3),
                                       name="Normalized MODIS Detection Probability"))
            fig_p.update_xaxes(type="log", title="Calibrated FRP (MW)")
            fig_p.update_yaxes(range=[0, 1.05], title="Detection Probability")
            fig_p.update_layout(title="Data-Driven Empirical MODIS Detection Curve")
            st.plotly_chart(style_fig(fig_p, 420), width="stretch")
            st.caption(f"Plateau (concurrent observation rate at high FRP): {curve.plateau:.2f} · n = {fmt_int(curve.n_events)} VIIRS events")
        else:
            st.info("Insufficient coincident pairs to fit an empirical detection curve; VIIRS-only events receive weight = 1.0.")

    lab = events_view["sensor_label"].value_counts().rename_axis("sensor_label").reset_index(name="events")
    pix = events_view.groupby("sensor_label")["merged_pixel_count"].mean().round(2).rename("avg_merged_pixels")
    lab = lab.merge(pix, on="sensor_label")
    fig_l = px.bar(lab, x="sensor_label", y="events", text="avg_merged_pixels", color="sensor_label",
                   color_discrete_map={hz.SENSOR_BOTH: EMBER, hz.SENSOR_VIIRS: C_VIIRS, hz.SENSOR_MODIS: C_MODIS})
    fig_l.update_traces(texttemplate="avg %{text} pixels/event", textposition="outside")
    fig_l.update_layout(title="Distribution of Harmonized Events by Sensor Label", showlegend=False, xaxis_title="")
    st.plotly_chart(style_fig(fig_l, 360), width="stretch")

# ---------------------------------------------------------------------------
# Tab 4 — Data Download Center
# ---------------------------------------------------------------------------
with tab_dl:
    out = hz.export_table(events_view)
    st.markdown(f"**Harmonized Hotspots Table** — {fmt_int(len(out))} records · {d0} to {d1} · {aoi_name}")
    st.dataframe(out.head(500), width="stretch", hide_index=True)
    st.download_button(
        "⬇️ Download koz_atlas_harmonized_hotspots.csv",
        out.to_csv(index=False).encode("utf-8"),
        file_name="koz_atlas_harmonized_hotspots.csv",
        mime="text/csv",
        type="primary",
    )
    c1, c2, c3 = st.columns(3)
    if not res.static_sites.empty:
        sites = res.static_sites.copy()
        for c in ("first_seen", "last_seen"):
            sites[c] = sites[c].dt.strftime("%Y-%m-%d")
        c1.download_button("Download Static Sources (CSV)", sites.to_csv(index=False).encode("utf-8"),
                           file_name="koz_atlas_static_sources.csv", mime="text/csv")
    raw_cols = ["point_id", "unified_hotspot_id", "latitude", "longitude", "utc_timestamp", "instrument", "satellite",
                "source", "frp", "brightness_k", "brightness_2_k", "scan", "track", "conf_class", "daynight", "is_static_source"]
    raw_out = points_view[raw_cols].copy()
    raw_out["utc_timestamp"] = raw_out["utc_timestamp"].dt.strftime("%Y-%m-%dT%H:%MZ")
    c2.download_button("Download Raw Pixel → Event Map (CSV)", raw_out.to_csv(index=False).encode("utf-8"),
                       file_name="koz_atlas_pixel_event_map.csv", mime="text/csv")
    meta = {
        "project": "KOZ Atlas",
        "team": "Kirvecanlar",
        "challenge": CHALLENGE_URL,
        "data_mode": mode,
        "aoi_name": aoi_name,
        "aoi_bbox_wsen": list(bbox),
        "date_range": [str(d0), str(d1)],
        "params": params,
        "static_sources_hidden": hide_static,
        "frp_calibration": res.calibration.describe(),
        "modis_detection_curve": {
            "frp_mw": (10 ** res.detection_curve.log_frp_centers).round(3).tolist(),
            "p_detect": res.detection_curve.p_detect.round(4).tolist(),
        },
        "stats": {k: v for k, v in s.items() if k != "label_counts"} | {"label_counts": s["label_counts"]},
    }
    c3.download_button("Download Run Metadata (JSON)", json.dumps(meta, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
                       file_name="koz_atlas_run_metadata.json", mime="application/json")

    with st.expander("CSV Column Dictionary"):
        st.markdown(
            """
| Column | Description |
|---|---|
| `unified_hotspot_id` | Harmonized Fire Event Identifier (`KOZ-YYYYMMDD-seq`) |
| `latitude`, `longitude` | FRP-weighted centroid of clustered observations |
| `utc_timestamp`, `utc_end` | Earliest and latest observation time within cluster (UTC) |
| `sensor_label` | `BOTH (MODIS+VIIRS)`, `VIIRS_CLUSTER_HARMONIZED`, or `MODIS_ONLY` |
| `merged_pixel_count` | Total raw pixels merged into event (`n_modis` + `n_viirs`) |
| `frp_modis_mw`, `frp_viirs_mw` | Sensor-specific FRP: sum within pass, mean across passes (MW) |
| `frp_viirs_calibrated_mw` | VIIRS FRP calibrated to MODIS scale via fitted regression |
| `calibrated_frp_mw` | Harmonized FRP: MODIS FRP if observed; otherwise calibrated VIIRS FRP |
| `modis_eq_weight` | Empirical MODIS detection probability (1.0 if observed by MODIS) |
| `modis_equivalent` | Boolean flag indicating `modis_eq_weight >= 0.5` |
| `is_static_source`, `event_class` | Static industrial thermal source flag and classification string |
"""
        )

st.markdown("---")
st.caption(
    f"KOZ Atlas · Team Kirvecanlar (Mehmet Halil KILIÇ, İbrahim Halil BİRLİK, Mirze KARA) · Batman University Computer Engineering · "
    f"Data: NASA FIRMS ([Area API Guide]({FIRMS_GUIDE_URL})) · [Challenge Description]({CHALLENGE_URL}) · [Live Application]({APP_URL}) · [GitHub]({REPO_URL})"
)
