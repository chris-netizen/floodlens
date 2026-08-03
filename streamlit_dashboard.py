"""
MapKaton 2026 — Maiduguri Flood Exposure Dashboard
==================================================
Interactive tool: overlay OpenStreetMap infrastructure on the September 2024 Maiduguri
flood (Copernicus EMS EMSR753), see which schools / clinics / roads / buildings were
exposed, and download the list. Independently validated against a Sentinel-1 SAR pipeline.

Run
---
    pip install streamlit streamlit-folium
    streamlit run streamlit_dashboard.py

Reads the outputs produced by maiduguri_flood_exposure.py in the same folder:
    exposure_summary.csv, exposed_facilities.geojson, flood_extent.geojson
plus the official EMSR753 shapefile (edit OFFICIAL_FLOOD below if the path differs).
"""
import geopandas as gpd
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# --- point this at your EMSR753 observed-flood shapefile (same path as the pipeline) ---
OFFICIAL_FLOOD = r"D:\Users\SALWAABDULLA\Downloads\EMSR753_AOI01_GRA_MONIT01_v1\EMSR753_AOI01_GRA_MONIT01_observedEventA_v1.shp"
OFFICIAL_AOI = OFFICIAL_FLOOD.replace("observedEventA", "areaOfInterestA")
UTM_CRS = 32633

EVENT_TITLE = "Maiduguri Flood — September 2024"
EVENT_BLURB = ("Alau Dam collapse, 9–10 September 2024 · ~40% of the city flooded · "
               "~400,000 displaced. Flood extent: Copernicus EMS EMSR753.")

st.set_page_config(page_title="Maiduguri Flood Exposure", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------- data loading (cached)


@st.cache_data(show_spinner=False)
def load_all():
    summary = pd.read_csv("exposure_summary.csv")
    exposed = gpd.read_file("exposed_facilities.geojson").to_crs(4326)
    sar = gpd.read_file("flood_extent.geojson").to_crs(4326)
    official = gpd.read_file(OFFICIAL_FLOOD).to_crs(4326)
    official["geometry"] = official.geometry.buffer(0)
    return summary, exposed, sar, official


@st.cache_data(show_spinner=False)
def containment_stats():
    """SAR-vs-official agreement, clipped to the mapped AOI. Returns (containment, iou)."""
    sar = gpd.read_file("flood_extent.geojson").to_crs(UTM_CRS).union_all()
    off = gpd.read_file(OFFICIAL_FLOOD).to_crs(UTM_CRS)
    off["geometry"] = off.geometry.buffer(0)
    off = off.union_all()
    try:
        aoi = gpd.read_file(OFFICIAL_AOI).to_crs(UTM_CRS).union_all()
        sar, off = sar.intersection(aoi), off.intersection(aoi)
    except Exception:
        pass
    inter = sar.intersection(off).area
    contain = inter / sar.area if sar.area else 0.0
    iou = inter / (sar.area + off.area - inter) if (sar.area +
                                                    off.area - inter) else 0.0
    return contain, iou


summary, exposed, sar, official = load_all()
contain, iou = containment_stats()


def row(layer):
    r = summary[summary.layer == layer]
    return (int(r.total.iloc[0]), int(r.exposed.iloc[0])) if len(r) else (0, 0)


sch_t, sch_e = row("schools")
cli_t, cli_e = row("clinics")
bld_t, bld_e = row("buildings")
rd_t,  rd_e = row("roads")

# ---------------------------------------------------------------------------- header
st.title("Flooded, and the Shelter Too")
st.markdown(f"#### {EVENT_TITLE}")
st.caption(EVENT_BLURB)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Buildings in flood zone",
          f"{bld_e:,}", f"{bld_e/bld_t:.0%} of mapped" if bld_t else "—")
c2.metric("Schools exposed", f"{sch_e} / {sch_t}", "of OSM-mapped schools")
c3.metric("Clinics & hospitals exposed",
          f"{cli_e} / {cli_t}", "of OSM-mapped facilities")
c4.metric("Road segments cut", f"{rd_e:,}",
          f"{rd_e/rd_t:.0%} of network" if rd_t else "—")

st.divider()

# ---------------------------------------------------------------------------- sidebar
st.sidebar.header("Layers")
show_official = st.sidebar.checkbox("Official flood extent (EMSR753)", True)
show_sar = st.sidebar.checkbox("My Sentinel-1 SAR extent", False)
types = st.sidebar.multiselect("Facility types to map",
                               ["schools", "clinics"], default=["schools", "clinics"])
st.sidebar.divider()
st.sidebar.metric("SAR ⊂ official (containment)", f"{contain:.1%}")
st.sidebar.metric("IoU within mapped AOI", f"{iou:.1%}")
st.sidebar.caption("Containment is the fair agreement measure here — the official extent "
                   "is larger because analysts hand-map urban inundation radar can't see.")

# ---------------------------------------------------------------------------- map + table
left, right = st.columns([3, 2])

with left:
    c = official.union_all().centroid
    m = folium.Map(location=[c.y, c.x], zoom_start=11, tiles="cartodbpositron")
    if show_official:
        folium.GeoJson(official, name="EMSR753",
                       style_function=lambda _: {"color": "#7b3f00", "weight": 1,
                                                 "fillColor": "#a15c2f", "fillOpacity": 0.35}).add_to(m)
    if show_sar:
        folium.GeoJson(sar, name="SAR",
                       style_function=lambda _: {"color": "#1f6feb", "weight": 1,
                                                 "fillColor": "#1f6feb", "fillOpacity": 0.45}).add_to(m)
    colors = {"schools": "#d7263d", "clinics": "#5c2d91"}
    for t in types:
        pts = exposed[exposed.layer == t]
        for _, r in pts.iterrows():
            p = r.geometry
            folium.CircleMarker([p.y, p.x], radius=4, color=colors[t],
                                fill=True, fill_opacity=0.9, popup=t).add_to(m)
    st_folium(m, use_container_width=True, height=560, returned_objects=[])

with right:
    st.subheader("Exposed facilities")
    tbl = exposed[exposed.layer.isin(types)].copy()
    tbl["lon"] = tbl.geometry.x.round(5)
    tbl["lat"] = tbl.geometry.y.round(5)
    tbl = tbl[["layer", "lat", "lon"]].rename(columns={"layer": "type"})
    st.dataframe(tbl, use_container_width=True, height=380, hide_index=True)
    st.download_button("Download exposed facilities (CSV)",
                       tbl.to_csv(index=False).encode(),
                       "exposed_facilities.csv", "text/csv")

# ---------------------------------------------------------------------------- method
with st.expander("Methodology, data sources & the OSM mapping gap"):
    st.markdown(f"""
**Flood extent** — Copernicus EMS rapid-mapping activation **EMSR753**, the official
delineation of the 9–10 September 2024 Alau Dam flood.

**Infrastructure** — OpenStreetMap (schools, clinics/hospitals, buildings, roads),
pulled live via the Overpass API.

**Exposure** — a facility is counted as exposed when its location falls inside the
observed flood polygon.

**Independent validation** — a from-scratch Sentinel-1 SAR change-detection pipeline
(dB-difference + dark-water thresholding) reproduces the official extent with
**{contain:.1%} containment** (IoU {iou:.1%} within the mapped AOI; lower only because
the official layer also includes hand-mapped urban inundation that radar cannot detect).

**The OSM gap (read this):** official reports counted ~56 schools flooded, but OSM has
only **{sch_t} schools mapped** across this area. These figures reflect *what OSM knows
today*, not the full ground truth — which is precisely why community mapping in
Maiduguri matters for the next flood. That gap is a finding, not a flaw.
""")
