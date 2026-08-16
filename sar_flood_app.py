"""
FloodLens — upload a Sentinel-1 SAR image, get the flood map and who it hit
===========================================================================
A no-code tool for remote-sensing users. Upload an analysis-ready VV backscatter
GeoTIFF of a flood event; the app automatically detects flooded vs dry areas, pulls
OpenStreetMap schools / clinics / roads over the scene, and reports what was exposed.

Run
---
    pip install streamlit streamlit-folium rasterio scikit-image geopandas osmnx folium shapely numpy
    streamlit run sar_flood_app.py
    # big files? raise the limit:  streamlit run sar_flood_app.py --server.maxUploadSize 800

INPUT (important): a *geocoded, calibrated* VV backscatter GeoTIFF — e.g. an ASF RTC
product, or a SNAP / Google Earth Engine export. A raw Copernicus .SAFE package will NOT
work: it is in radar geometry with uncalibrated values and needs terrain correction first.
Optionally upload a pre-flood image of the same area to switch on change detection
(which removes permanent rivers and lakes automatically).
"""
# --- Use the OS certificate store for TLS. Some machines run antivirus (e.g. Avast)
#     or a corporate proxy that intercepts HTTPS and presents its own root cert; that
#     root is trusted by Windows but NOT by Python's bundled certifi list, so Overpass
#     calls fail SSL verification and OSM silently returns 0 features. truststore makes
#     Python trust whatever the OS trusts, fixing it without disabling the scanner. ---
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import streamlit as st
from streamlit_folium import st_folium
import folium
import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, shape
from skimage.filters import threshold_otsu
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.features import shapes as rio_shapes
import rasterio
import numpy as np
import tempfile
import pyproj
import os
# --- Force the correct PROJ database: a PostgreSQL/PostGIS install on this machine ships
#     an older proj.db that otherwise shadows rasterio's and breaks EPSG lookups. ---
for _v in ("PROJ_LIB", "PROJ_DATA"):
    os.environ.pop(_v, None)
os.environ["PROJ_LIB"] = os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()


st.set_page_config(page_title="FloodLens — SAR flood exposure", layout="wide")

OSM_TAGS = {"schools": {"amenity": "school"},
            "clinics": {"amenity": ["clinic", "hospital", "doctors"], "healthcare": True}}
UTM_HINT = None  # auto-picked per scene


# ------------------------------------------------------------------ raster helpers
def _read_band(path, band):
    with rasterio.open(path) as src:
        arr = src.read(band).astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.transform, src.crs, src.bounds, src.count


def to_db(a):
    """Ensure a decibel-like scale. If values look linear (all >= 0), log-scale them.
    Any monotonic transform preserves the water/land separation Otsu relies on."""
    a = a.astype("float32")
    finite = a[np.isfinite(a)]
    if finite.size and np.nanmin(finite) >= 0:
        a = np.where(a > 0, 10 * np.log10(a + 1e-6), np.nan)
    return a


def align_to(ref_shape, ref_transform, ref_crs, src_arr, src_transform, src_crs):
    """Put the pre image on the during grid. If they already share a grid (same source /
    export), no reprojection is needed — which also sidesteps rasterio's PROJ."""
    if src_arr.shape == ref_shape and src_transform == ref_transform:
        return src_arr
    try:
        dst = np.full(ref_shape, np.nan, dtype="float32")
        reproject(source=src_arr, destination=dst,
                  src_transform=src_transform, src_crs=src_crs,
                  dst_transform=ref_transform, dst_crs=ref_crs,
                  resampling=Resampling.bilinear)
        return dst
    except Exception as e:
        raise RuntimeError("Couldn't align the pre-flood image to the during image. Use a "
                           "pre-flood image with the same footprint/resolution, or run "
                           "with the during image only.") from e


# ------------------------------------------------------------------ flood detection
def detect_flood(during_db, pre_db, sensitivity, transform, crs, min_area_m2):
    """Otsu water threshold on the during image; if a pre image is supplied, also require
    the pixel to have darkened (change detection) to drop permanent water."""
    valid = np.isfinite(during_db)
    t = threshold_otsu(during_db[valid]) + sensitivity      # dark side = water
    water = (during_db < t) & valid

    if pre_db is not None:
        diff = during_db - pre_db
        # must be darker than before
        water &= np.isfinite(diff) & (diff < 0)

    # vectorize
    mask = water.astype("uint8")
    polys = [shape(g) for g, v in rio_shapes(
        mask, mask=water, transform=transform) if v == 1]
    if not polys:
        return gpd.GeoDataFrame(geometry=[], crs=crs), t
    gdf = gpd.GeoDataFrame(geometry=polys, crs=crs)

    # drop specks by area in a metric CRS
    utm = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm)
    gdf = gdf[gdf.area >= min_area_m2].to_crs(4326)
    return gdf, t


# ------------------------------------------------------------------ OSM + exposure
@st.cache_data(show_spinner=False)
def fetch_osm(bounds_wgs):
    left, bottom, right, top = bounds_wgs
    poly = box(left, bottom, right, top)
    layers = {}
    errors = []
    for name, tags in OSM_TAGS.items():
        try:
            g = ox.features_from_polygon(poly, tags)
            g = g[~g.geometry.is_empty & g.geometry.notna()]
            layers[name] = g.to_crs(4326)
        except Exception as e:
            layers[name] = gpd.GeoDataFrame(geometry=[], crs=4326)
            errors.append(f"{name}: {type(e).__name__}: {e}")
    try:
        roads = ox.graph_to_gdfs(ox.graph_from_polygon(
            poly, network_type="drive"), nodes=False)
        layers["roads"] = roads.to_crs(4326)
    except Exception as e:
        layers["roads"] = gpd.GeoDataFrame(geometry=[], crs=4326)
        errors.append(f"roads: {type(e).__name__}: {e}")
    return layers, errors


def exposure(layers, flood_gdf):
    if flood_gdf.empty:
        return {k: (len(v), 0) for k, v in layers.items()}, gpd.GeoDataFrame(geometry=[], crs=4326)
    fu = flood_gdf.union_all()
    rows, hits = {}, []
    for name, gdf in layers.items():
        if gdf.empty:
            rows[name] = (0, 0)
            continue
        pts = gdf.copy()
        pts["geometry"] = pts.geometry.representative_point()
        hit = pts[pts.geometry.within(fu)]
        rows[name] = (len(gdf), len(hit))
        if len(hit) and name in ("schools", "clinics"):
            h = hit.copy()
            h["layer"] = name
            hits.append(h[["layer", "geometry"]])
    exposed = gpd.GeoDataFrame(pd.concat(hits, ignore_index=True), crs=4326) if hits \
        else gpd.GeoDataFrame(geometry=[], crs=4326)
    return rows, exposed


# ------------------------------------------------------------------ UI
st.title("FloodLens")
st.caption("Upload a Sentinel-1 SAR image of a flood. Get the flooded area and the schools, "
           "clinics, and roads it hit — automatically.")

with st.sidebar:
    st.header("1 · Upload SAR")
    during_f = st.file_uploader(
        "During-flood VV GeoTIFF (required) — a sample (sample_maiduguri_vv.tif) is in the GitHub repo", type=["tif", "tiff"])
    pre_f = st.file_uploader("Pre-flood VV GeoTIFF (optional — enables change detection)",
                             type=["tif", "tiff"])
    band = st.number_input("VV band number", 1, 4, 1)
    st.header("2 · Detection")
    sensitivity = st.slider("Sensitivity (dB offset)", -5.0, 5.0, 0.0, 0.5,
                            help="Nudge the automatic water threshold. + catches more, − is stricter.")
    min_area = st.number_input("Min flood patch (m²)", 100, 50000, 2000, 100)
    st.caption("Input must be a geocoded, calibrated VV GeoTIFF (e.g. ASF RTC, or a "
               "SNAP/GEE export) — not a raw .SAFE package.")

if during_f is None:
    st.info("⬅️ Upload a during-flood VV GeoTIFF to begin. Add a pre-flood image too for "
            "the cleanest result — change detection automatically ignores permanent rivers "
            "and lakes.")
    st.stop()

# read during (+ pre)
with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
    tmp.write(during_f.getbuffer())
    during_path = tmp.name
d_arr, d_tr, d_crs, d_bounds, d_count = _read_band(during_path, band)
during_db = to_db(d_arr)

pre_db = None
if pre_f is not None:
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(pre_f.getbuffer())
        pre_path = tmp.name
    p_arr, p_tr, p_crs, _, _ = _read_band(pre_path, band)
    pre_db = align_to(during_db.shape, d_tr, d_crs, to_db(p_arr), p_tr, p_crs)

with st.spinner("Detecting flooded area…"):
    flood_gdf, thr = detect_flood(
        during_db, pre_db, sensitivity, d_tr, d_crs.to_wkt(), min_area)

if d_crs is None:
    st.error("This GeoTIFF has no coordinate system (CRS). Upload a geocoded image "
             "(e.g. an ASF RTC product or a SNAP/GEE export).")
    st.stop()
# Reproject the raster bounds to WGS84 via pyproj/geopandas (rasterio's PROJ may be shadowed).
_b = gpd.GeoSeries([box(*d_bounds)], crs=d_crs.to_wkt()
                   ).to_crs(4326).total_bounds
bounds_wgs = (float(_b[0]), float(_b[1]), float(_b[2]), float(_b[3]))
with st.spinner("Fetching OpenStreetMap infrastructure over the scene…"):
    layers, osm_errors = fetch_osm(tuple(bounds_wgs))
if osm_errors:
    st.warning(
        "Could not reach OpenStreetMap (Overpass API) — infrastructure counts may be 0. "
        "This is usually a network/SSL issue (antivirus or proxy HTTPS scanning), not the "
        "flood sensitivity. Details:\n\n" + "\n\n".join(osm_errors))
rows, exposed = exposure(layers, flood_gdf)

# metrics
mode = "change detection (pre + during)" if pre_db is not None else "single-image Otsu"
st.success(f"Flood detected · method: {mode} · water threshold ≈ {thr:.1f} dB")
m1, m2, m3 = st.columns(3)
m1.metric("Schools exposed", f"{rows['schools'][1]} / {rows['schools'][0]}")
m2.metric("Clinics & hospitals exposed",
          f"{rows['clinics'][1]} / {rows['clinics'][0]}")
m3.metric("Road segments cut", f"{rows['roads'][1]:,}")
st.caption(
    "These counts come from FloodLens's **live SAR detection**, which is deliberately "
    "conservative in dense urban areas (radar can't see water between buildings). The "
    "Maiduguri case study reports higher headline figures because it uses the official, "
    "hand-refined Copernicus EMSR753 extent — see the README. Both are the same tool, two "
    "flood sources, honestly compared."
)

# map
left, right = st.columns([3, 2])
with left:
    c = box(*bounds_wgs).centroid
    fmap = folium.Map(location=[c.y, c.x],
                      zoom_start=11, tiles="cartodbpositron")
    if not flood_gdf.empty:
        folium.GeoJson(flood_gdf, name="flood",
                       style_function=lambda _: {"color": "#1f6feb", "weight": 1,
                                                 "fillColor": "#1f6feb", "fillOpacity": 0.4}).add_to(fmap)
    colors = {"schools": "#d7263d", "clinics": "#5c2d91"}
    for _, r in exposed.iterrows():
        p = r.geometry
        folium.CircleMarker([p.y, p.x], radius=4, color=colors.get(r.layer, "#333"),
                            fill=True, fill_opacity=0.9, popup=r.layer).add_to(fmap)
    st_folium(fmap, use_container_width=True, height=560, returned_objects=[])

with right:
    st.subheader("Exposed facilities")
    if exposed.empty:
        st.write("No mapped schools or clinics fell inside the detected flood.")
    else:
        t = exposed.copy()
        t["lat"] = t.geometry.y.round(5)
        t["lon"] = t.geometry.x.round(5)
        t = t[["layer", "lat", "lon"]].rename(columns={"layer": "type"})
        st.dataframe(t, use_container_width=True, height=340, hide_index=True)
        st.download_button("Download exposed facilities (CSV)",
                           t.to_csv(index=False).encode(), "exposed_facilities.csv", "text/csv")
    if not flood_gdf.empty:
        st.download_button("Download flood extent (GeoJSON)",
                           flood_gdf.to_json().encode(), "flood_extent.geojson", "application/geo+json")

with st.expander("How this works & limits"):
    st.markdown("""
**Detection.** SAR sees smooth open water as dark (it reflects radar away). The app finds
the dark-vs-bright split automatically with **Otsu thresholding** — no manual tuning. If you
also upload a **pre-flood** image, it additionally requires each pixel to have *darkened*
since before, which removes permanent rivers, lakes, and other always-dark surfaces.

**Exposure.** Schools, clinics, and roads are pulled live from **OpenStreetMap** over the
image footprint; a facility counts as exposed when it lies inside the detected flood.

**Limits.** Radar struggles to see shallow water between dense buildings, so urban cores may
be under-detected — upload the pre-flood image and adjust *Sensitivity* if needed. Exposure
counts reflect what OpenStreetMap has mapped, not full ground truth. Input must be geocoded,
calibrated VV backscatter (ASF RTC / SNAP / GEE export), not a raw .SAFE package.
""")
