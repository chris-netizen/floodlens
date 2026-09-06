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
from scipy import ndimage as ndi
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.features import shapes as rio_shapes, rasterize
import rasterio
import numpy as np
import tempfile
import pyproj
import os
import json
import math
from datetime import date, timedelta
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
def despeckle(db, size):
    """Median-filter a dB image to suppress SAR speckle, preserving the NaN mask.
    size <= 1 disables it. NaNs are filled with the scene median during filtering,
    then restored, so nodata borders don't smear into the result."""
    if size is None or size <= 1:
        return db
    m = np.isfinite(db)
    if not m.any():
        return db
    filled = np.where(m, db, np.nanmedian(db))
    smoothed = ndi.median_filter(filled, size=int(size))
    return np.where(m, smoothed, np.nan)


def detect_flood(during_db, pre_db, sensitivity, transform, crs, min_area_m2,
                 speckle_win=5, change_drop=2.0, water_ceiling=-10.0,
                 hand_arr=None, hand_max=15.0):
    """Otsu water threshold on a despeckled during image; if a pre image is supplied,
    also require the pixel to have *meaningfully* darkened (change detection) to drop
    permanent water and seasonal land change.

    Hardening vs. the naive version:
      • speckle_win  — median despeckle first, so single-pixel noise can't pass.
      • change_drop  — a pixel must have darkened by >= this many dB (not just any
                       darkening), which removes monsoon farmland/soil-moisture change.
      • water_ceiling— caps the auto threshold so damp land can't be read as water when
                       Otsu drifts high (matters most in single-image mode).
      • hand_arr/max — Height Above Nearest Drainage mask: drop pixels sitting more than
                       hand_max metres above the nearest river (they can't physically
                       flood), removing steep-terrain radar shadow/layover artifacts.
      • morphological opening + area filter remove residual isolated specks.
    """
    during_s = despeckle(during_db, speckle_win)
    valid = np.isfinite(during_s)
    t = threshold_otsu(during_s[valid]) + sensitivity       # dark side = water
    if water_ceiling is not None:
        t = min(t, float(water_ceiling))
    water = (during_s < t) & valid

    if pre_db is not None:
        pre_s = despeckle(pre_db, speckle_win)
        diff = during_s - pre_s
        # must have darkened by a meaningful margin, not just any amount
        water &= np.isfinite(diff) & (diff < -abs(change_drop))

    # terrain mask: only pixels low enough above drainage can flood
    if hand_arr is not None:
        water &= np.isfinite(hand_arr) & (hand_arr <= float(hand_max))

    # break isolated-pixel speckle bridges before vectorizing
    if water.any():
        water = ndi.binary_opening(water, structure=np.ones((3, 3), bool))

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


# ------------------------------------------------------------------ Earth Engine auto-fetch
#   Lets non-GIS users skip the "go find an analysis-ready GeoTIFF" step entirely: pick a
#   place + a flood date, and we pull Sentinel-1 VV straight from Google Earth Engine. The
#   COPERNICUS/S1_GRD VV band is already calibrated sigma-0 in dB, so it feeds the same
#   detect_flood() path as an uploaded ASF/SNAP scene.
S1 = "COPERNICUS/S1_GRD"


def _write_tif(data):
    """Write GeoTIFF bytes to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(data)
        return tmp.name


def init_ee():
    """Initialise Earth Engine. Returns (ok, detail).
    Order: service account from st.secrets (Streamlit Cloud) → local user credentials.
    Success is remembered for the session; failures are not cached, so a retry works
    once the user sets EE_PROJECT / adds secrets."""
    if st.session_state.get("_ee_ready"):
        return True, "initialised"
    import ee
    # 1) service-account JSON in secrets — the deployment path
    try:
        has_sa = "ee_service_account" in st.secrets
    except Exception:
        has_sa = False
    if has_sa:
        try:
            info = dict(st.secrets["ee_service_account"])
            creds = ee.ServiceAccountCredentials(
                info["client_email"], key_data=json.dumps(info))
            ee.Initialize(creds, project=info.get("project_id"))
            st.session_state["_ee_ready"] = True
            return True, "service account"
        except Exception as e:
            return False, f"service-account init failed: {type(e).__name__}: {e}"
    # 2) local persistent credentials (after `earthengine authenticate`)
    project = None
    try:
        project = st.secrets.get("EE_PROJECT")
    except Exception:
        pass
    project = project or os.environ.get("EE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    try:
        ee.Initialize(project=project)
        st.session_state["_ee_ready"] = True
        return True, "user credentials"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


@st.cache_data(show_spinner=False)
def geocode_place(place):
    """Place name -> (lat, lon) via OSM Nominatim (through OSMnx)."""
    lat, lon = ox.geocode(place)
    return float(lat), float(lon)


def bbox_km(lat, lon, km):
    """Square AOI of side `km` centred on (lat, lon), returned as (w, s, e, n) degrees."""
    dlat = (km / 111.0) / 2.0
    dlon = (km / (111.0 * max(math.cos(math.radians(lat)), 1e-6))) / 2.0
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


@st.cache_data(show_spinner=False)
def fetch_s1_vv(aoi_bounds, start, end, scale, orbit):
    """Median Sentinel-1 VV (dB) over the AOI/date window, downloaded as a GeoTIFF (bytes).
    Returns (tif_bytes, n_scenes). Raises RuntimeError if no scenes match."""
    import ee
    import requests
    w, s, e, n = aoi_bounds
    aoi = ee.Geometry.Rectangle([w, s, e, n])
    col = (ee.ImageCollection(S1)
           .filterBounds(aoi)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filterDate(start, end)
           .select("VV"))
    if orbit in ("ASCENDING", "DESCENDING"):
        col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit))
    n_scenes = int(col.size().getInfo())
    if n_scenes == 0:
        raise RuntimeError(
            f"No Sentinel-1 VV scenes for this area between {start} and {end}. "
            "Try a wider window, a different date, or the other orbit direction.")
    img = col.median().clip(aoi).toFloat()
    url = img.getDownloadURL({
        "scale": scale, "region": aoi, "format": "GEO_TIFF", "crs": "EPSG:4326"})
    r = requests.get(url, timeout=240)
    r.raise_for_status()
    return r.content, n_scenes


@st.cache_data(show_spinner=False)
def fetch_s2_water(aoi_bounds, start, end, scale):
    """Independent optical check. Returns a 2-band GeoTIFF (bytes): band 1 = clear-sky
    mask (1 where cloud/shadow-free), band 2 = water mask (1 where MNDWI>0 in clear sky),
    plus the number of Sentinel-2 scenes used. Cloud + shadow are removed with Google's
    Cloud Score+; MNDWI (green-SWIR) handles muddy floodwater better than NDWI. Because
    flood debris/mud persists for days, a *later* clear pass still maps it even when the
    flood day was fully clouded."""
    import ee
    import requests
    w, s, e, n = aoi_bounds
    aoi = ee.Geometry.Rectangle([w, s, e, n])
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi).filterDate(start, end)
          .linkCollection(ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"), ["cs"]))
    n_scenes = int(s2.size().getInfo())
    if n_scenes == 0:
        raise RuntimeError(f"No Sentinel-2 scenes between {start} and {end}.")
    comp = s2.map(lambda i: i.updateMask(i.select("cs").gte(0.6))).median()
    clear = comp.select("B3").mask().rename("clear")
    water = comp.normalizedDifference(["B3", "B11"]).gt(0.0).And(clear).rename("water")
    out = clear.addBands(water).toByte().clip(aoi)
    url = out.getDownloadURL({
        "scale": scale, "region": aoi, "format": "GEO_TIFF", "crs": "EPSG:4326"})
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return r.content, n_scenes


@st.cache_data(show_spinner=False)
def fetch_hand(aoi_bounds, scale):
    """Height Above Nearest Drainage (m) from MERIT Hydro, as a GeoTIFF (bytes). A pixel's
    HAND is how high it sits above the river it drains to; pixels far above any drainage
    can't physically flood, so they're used to mask out steep-terrain SAR artifacts."""
    import ee
    import requests
    w, s, e, n = aoi_bounds
    aoi = ee.Geometry.Rectangle([w, s, e, n])
    hnd = ee.Image("MERIT/Hydro/v1_0_1").select("hnd").clip(aoi).toFloat()
    url = hnd.getDownloadURL({
        "scale": scale, "region": aoi, "format": "GEO_TIFF", "crs": "EPSG:4326"})
    r = requests.get(url, timeout=240)
    r.raise_for_status()
    return r.content


# ------------------------------------------------------------------ UI
st.title("FloodLens")
st.caption("Pick a place and flood date — or upload a Sentinel-1 SAR image. Get the flooded "
           "area and the schools, clinics, and roads it hit — automatically.")

with st.sidebar:
    st.header("1 · Data source")
    source = st.radio(
        "How do you want to provide the SAR image?",
        ["Auto-fetch (Earth Engine)", "Upload GeoTIFF"],
        help="Auto-fetch pulls Sentinel-1 straight from Earth Engine — no GIS software or "
             "manual downloads needed. Upload if you already have an analysis-ready scene.")

    band = 1
    during_f = pre_f = None
    place = flood_date = orbit = None
    window_days = aoi_km = scale_m = None
    use_change = True
    validate_optical = False
    opt_days = 21
    use_hand = False
    hand_max = 15

    if source == "Upload GeoTIFF":
        during_f = st.file_uploader(
            "During-flood VV GeoTIFF (required) — sample (sample_maiduguri_vv.tif) is in the repo",
            type=["tif", "tiff"])
        pre_f = st.file_uploader("Pre-flood VV GeoTIFF (optional — enables change detection)",
                                 type=["tif", "tiff"])
        band = st.number_input("VV band number", 1, 4, 1)
    else:
        place = st.text_input("Place or area", "Maiduguri, Nigeria",
                              help="A city, town, or region name — geocoded via OpenStreetMap.")
        flood_date = st.date_input("Approximate flood date", value=date(2024, 9, 10))
        window_days = st.slider("Search window (± days)", 6, 36, 12, 2,
                                help="Sentinel-1 revisits every ~6–12 days. Widen if no scene is found.")
        aoi_km = st.slider("Area size (km across)", 5, 40, 20, 5)
        with st.expander("Advanced fetch options"):
            use_change = st.checkbox("Use a pre-flood baseline (change detection)", True,
                                     help="Removes permanent rivers/lakes by requiring pixels to have darkened.")
            scale_m = st.select_slider("Resolution (m/pixel)", [10, 20, 30], 20)
            orbit = st.selectbox("Orbit direction", ["Any", "ASCENDING", "DESCENDING"])
            use_hand = st.checkbox(
                "Terrain mask (HAND)", True,
                help="Drop pixels too high above the nearest river to flood — removes "
                     "steep-terrain radar shadow/layover artifacts. Harmless on flat areas.")
            hand_max = st.slider(
                "Max height above drainage (m)", 2, 40, 15,
                help="Pixels more than this many metres above the nearest drainage are masked out.")
            validate_optical = st.checkbox(
                "Validate with optical (Sentinel-2)", True,
                help="Independently confirm the SAR flood against optical water on cloud-free "
                     "pixels. Because flood mud/debris persists, a later clear pass still maps it.")
            opt_days = st.slider(
                "Optical look-ahead (days after flood)", 3, 45, 21,
                help="How long after the flood to search for a clear Sentinel-2 pass. Longer = "
                     "more chance of clear sky as monsoon clouds lift, but water may recede.")

    st.header("2 · Detection")
    sensitivity = st.slider("Sensitivity (dB offset)", -5.0, 5.0, 0.0, 0.5,
                            help="Nudge the automatic water threshold. + catches more, − is stricter.")
    min_area = st.number_input("Min flood patch (m²)", 100, 50000, 2000, 100)
    with st.expander("Advanced detection (noise control)"):
        speckle_win = st.select_slider(
            "Speckle filter window (px)", [1, 3, 5, 7], 5,
            help="Median despeckle before thresholding. 1 = off. Larger = smoother, fewer specks.")
        change_drop = st.slider(
            "Min backscatter drop for change (dB)", 0.0, 6.0, 2.0, 0.5,
            help="With a pre-flood baseline, a pixel must have darkened by at least this much "
                 "to count as flood — filters out seasonal farmland/soil-moisture change.")
        water_ceiling = st.slider(
            "Max water threshold (dB)", -18.0, -6.0, -10.0, 1.0,
            help="Caps the auto threshold so damp land isn't read as water if Otsu drifts high.")
    if source == "Upload GeoTIFF":
        st.caption("Input must be a geocoded, calibrated VV GeoTIFF (e.g. ASF RTC, or a "
                   "SNAP/GEE export) — not a raw .SAFE package.")

# ---- resolve the input to a during (+ optional pre) GeoTIFF path ----
during_path = pre_path = None

if source == "Upload GeoTIFF":
    if during_f is None:
        st.info("⬅️ Upload a during-flood VV GeoTIFF to begin. Add a pre-flood image too for "
                "the cleanest result — change detection automatically ignores permanent rivers "
                "and lakes.")
        st.stop()
    during_path = _write_tif(during_f.getbuffer())
    if pre_f is not None:
        pre_path = _write_tif(pre_f.getbuffer())
else:
    ok, detail = init_ee()
    if not ok:
        st.error("Earth Engine isn't set up on this deployment, so auto-fetch is unavailable. "
                 "Switch to **Upload GeoTIFF**, or configure Earth Engine (see below).")
        st.caption(f"Details: {detail}")
        with st.expander("How to enable auto-fetch"):
            st.markdown(
                "**Local:** `pip install earthengine-api`, then run `earthengine authenticate` "
                "and set your Cloud project via the `EE_PROJECT` env var.\n\n"
                "**Streamlit Cloud:** add a Google Cloud **service-account** key to the app's "
                "*Secrets* as a `[ee_service_account]` table (the JSON key's fields), with Earth "
                "Engine enabled for that project.")
        st.stop()
    if not (place or "").strip():
        st.info("⬅️ Enter a place or area to fetch Sentinel-1 for.")
        st.stop()
    try:
        lat, lon = geocode_place(place)
    except Exception:
        st.error(f"Couldn't find “{place}” on OpenStreetMap. Try a more specific name "
                 "(e.g. add the country).")
        st.stop()
    aoi = bbox_km(lat, lon, aoi_km)
    orbit_f = orbit if orbit in ("ASCENDING", "DESCENDING") else None
    post_start = (flood_date - timedelta(days=3)).isoformat()
    post_end = (flood_date + timedelta(days=window_days)).isoformat()
    pre_end = (flood_date - timedelta(days=30)).isoformat()
    pre_start = (flood_date - timedelta(days=30 + max(window_days, 24))).isoformat()
    try:
        with st.spinner(f"Fetching Sentinel-1 over {place} from Earth Engine…"):
            d_bytes, n_post = fetch_s1_vv(aoi, post_start, post_end, scale_m, orbit_f)
            during_path = _write_tif(d_bytes)
            n_pre = 0
            if use_change:
                p_bytes, n_pre = fetch_s1_vv(aoi, pre_start, pre_end, scale_m, orbit_f)
                pre_path = _write_tif(p_bytes)
    except Exception as e:
        st.error(f"Earth Engine fetch failed: {type(e).__name__}: {e}")
        st.stop()
    st.success(
        f"Fetched Sentinel-1 VV over **{place}** ({lat:.3f}, {lon:.3f}): {n_post} during-scene(s)"
        + (f" + {n_pre} pre-flood scene(s)" if use_change else "")
        + f", composited at {scale_m} m/pixel.")

# ---- shared: read the resolved raster(s) into the detection pipeline ----
d_arr, d_tr, d_crs, d_bounds, d_count = _read_band(during_path, band)
during_db = to_db(d_arr)

pre_db = None
if pre_path is not None:
    p_arr, p_tr, p_crs, _, _ = _read_band(pre_path, band)
    pre_db = align_to(during_db.shape, d_tr, d_crs, to_db(p_arr), p_tr, p_crs)

# terrain (HAND) mask, aligned to the during grid — auto-fetch only
hand_arr = None
if source != "Upload GeoTIFF" and use_hand:
    try:
        with st.spinner("Fetching terrain (Height Above Nearest Drainage)…"):
            h_bytes = fetch_hand(aoi, scale_m)
        with rasterio.open(_write_tif(h_bytes)) as hsrc:
            h_src = hsrc.read(1).astype("float32")
            if hsrc.nodata is not None:
                h_src[h_src == hsrc.nodata] = np.nan
            h_tr, h_crs = hsrc.transform, hsrc.crs
        hand_arr = np.full(during_db.shape, np.nan, dtype="float32")
        reproject(source=h_src, destination=hand_arr,
                  src_transform=h_tr, src_crs=h_crs,
                  dst_transform=d_tr, dst_crs=d_crs, resampling=Resampling.bilinear)
    except Exception as e:
        st.warning(f"Terrain (HAND) mask unavailable — proceeding without it: "
                   f"{type(e).__name__}: {e}")
        hand_arr = None

with st.spinner("Detecting flooded area…"):
    flood_gdf, thr = detect_flood(
        during_db, pre_db, sensitivity, d_tr, d_crs.to_wkt(), min_area,
        speckle_win=speckle_win, change_drop=change_drop, water_ceiling=water_ceiling,
        hand_arr=hand_arr, hand_max=hand_max)

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

# ---- optional: independent optical validation (auto-fetch only) ----
optical_ready = False
opt_overlay = opt_img_bounds = opt_note = None
if source != "Upload GeoTIFF" and validate_optical:
    o_start = flood_date.isoformat()
    o_end = (flood_date + timedelta(days=opt_days)).isoformat()
    try:
        with st.spinner("Fetching Sentinel-2 optical for validation…"):
            s2_bytes, n_s2 = fetch_s2_water(aoi, o_start, o_end, scale_m)
        with rasterio.open(_write_tif(s2_bytes)) as s2src:
            clear_arr = s2src.read(1)
            water_arr = s2src.read(2)
            o_transform, o_shape, o_b = s2src.transform, (s2src.height, s2src.width), s2src.bounds
        clear_b = clear_arr > 0
        opt_water = (water_arr > 0) & clear_b
        coverage = float(clear_b.mean()) if clear_b.size else 0.0
        px_km2 = (scale_m * scale_m) / 1e6
        if coverage < 0.10:
            opt_note = ("warn",
                        f"⚠️ Too cloudy to validate: only {coverage*100:.0f}% of the scene was "
                        f"cloud-free in the {opt_days}-day window ({n_s2} Sentinel-2 scenes). "
                        "Optical confirmation withheld — try a longer look-ahead.")
        else:
            sar_ras = np.zeros(o_shape, dtype="uint8")
            if not flood_gdf.empty:
                sar_ras = rasterize(((g, 1) for g in flood_gdf.geometry),
                                    out_shape=o_shape, transform=o_transform, fill=0, dtype="uint8")
            sar_b = (sar_ras > 0) & clear_b
            confirmed, sar_clear_n = int((sar_b & opt_water).sum()), int(sar_b.sum())
            optical_ready = True
            opt_img_bounds = [[o_b.bottom, o_b.left], [o_b.top, o_b.right]]
            opt_overlay = np.zeros((o_shape[0], o_shape[1], 4), dtype="uint8")
            opt_overlay[opt_water] = (0, 200, 255, 160)   # cyan where optical water
            if sar_clear_n:
                opt_note = ("ok",
                            f"**Optical validation** · {coverage*100:.0f}% cloud-free "
                            f"({n_s2} S2 scenes). **{100*confirmed/sar_clear_n:.0f}% of clear-sky "
                            f"SAR flood is confirmed as water by Sentinel-2** "
                            f"({confirmed*px_km2:.1f} of {sar_clear_n*px_km2:.1f} km²). "
                            "Cyan overlay = optical water.")
            else:
                opt_note = ("info",
                            f"**Optical validation** · {coverage*100:.0f}% cloud-free "
                            f"({n_s2} S2 scenes). No SAR flood fell in clear-sky pixels to "
                            "cross-check; optical water (cyan) is shown for reference.")
    except Exception as e:
        opt_note = ("warn", f"Optical validation unavailable: {type(e).__name__}: {e}")

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

if opt_note:
    {"ok": st.success, "warn": st.warning, "info": st.info}[opt_note[0]](opt_note[1])

# map
left, right = st.columns([3, 2])
with left:
    c = box(*bounds_wgs).centroid
    # OpenStreetMap tiles need no API key; CARTO basemaps now require one.
    fmap = folium.Map(location=[c.y, c.x], zoom_start=11,
                      tiles="OpenStreetMap")
    if not flood_gdf.empty:
        folium.GeoJson(flood_gdf, name="SAR flood",
                       style_function=lambda _: {"color": "#1f6feb", "weight": 1,
                                                 "fillColor": "#1f6feb", "fillOpacity": 0.4}).add_to(fmap)
    if optical_ready and opt_overlay is not None:
        folium.raster_layers.ImageOverlay(
            image=opt_overlay, bounds=opt_img_bounds, opacity=0.6,
            name="Optical water (Sentinel-2)", interactive=False).add_to(fmap)
    colors = {"schools": "#d7263d", "clinics": "#5c2d91"}
    for _, r in exposed.iterrows():
        p = r.geometry
        folium.CircleMarker([p.y, p.x], radius=4, color=colors.get(r.layer, "#333"),
                            fill=True, fill_opacity=0.9, popup=r.layer).add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
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

**Optical validation (auto-fetch).** As an independent check, FloodLens can pull **Sentinel-2**
optical over the same area, mask cloud and shadow with **Cloud Score+**, and map water with
**MNDWI**. On the cloud-free pixels it reports what fraction of the SAR flood is optically
confirmed as water (cyan overlay). Because flood mud and debris *persist*, a clear pass in the
days/weeks after the event still maps the impact even when the flood day itself was overcast —
and when too little of the scene is clear, the tool says so rather than reporting a weak number.

**Getting the image.** *Auto-fetch* pulls Sentinel-1 VV straight from Google Earth Engine for
the place and date you choose (a median composite over a short window, with a dry pre-flood
baseline for change detection) — no GIS software or manual downloads. The `COPERNICUS/S1_GRD`
VV band is already calibrated backscatter in dB, the same quantity a terrain-corrected upload
provides. Prefer *Upload* when you have your own analysis-ready scene (ASF RTC / SNAP export).

**Terrain masking (auto-fetch).** In hilly or mountainous scenes, radar shadow and layover
on slopes can masquerade as water. FloodLens masks these with **HAND** (Height Above Nearest
Drainage, from MERIT Hydro): any pixel sitting more than a set height above the nearest river
can't physically flood, so it's excluded. This is harmless on flat terrain and removes most
false positives on steep terrain.

**Limits.** Radar struggles to see shallow water between dense buildings, so urban cores may
be under-detected — keep the pre-flood baseline on and adjust *Sensitivity* if needed. Exposure
counts reflect what OpenStreetMap has mapped, not full ground truth. Uploaded input must be
geocoded, calibrated VV backscatter (ASF RTC / SNAP / GEE export), not a raw .SAFE package.
""")
