"""
MapKaton 2026 — Flood-exposure of public infrastructure, Maiduguri (Alau Dam, Sep 2024)
=======================================================================================
Reproducible spine: Sentinel-1 SAR flood extent (Google Earth Engine) x OpenStreetMap
infrastructure (schools, clinics/hospitals, roads, buildings) -> exposure counts + CSV.

This is the analysis core. The Streamlit dashboard consumes the outputs it writes.

Setup
-----
    python -m venv .venv && source .venv/bin/activate
    pip install earthengine-api geemap osmnx geopandas folium shapely mapclassify rasterio
    earthengine authenticate      # one-time, uses your GEE account

Run
---
    python maiduguri_flood_exposure.py
    # writes: flood_extent.geojson, exposed_facilities.csv, exposure_map.html

Judging note: keep every threshold and date in the CONFIG block below so a reviewer
can reproduce the exact numbers. That is 15% of your score (reproducibility) that most
entrants leave on the table.
"""

import ee
import geemap
import osmnx as ox
import geopandas as gpd
import pandas as pd
import sys
import rasterio
from rasterio.features import shapes as rio_shapes

# ----------------------------------------------------------------------------------
# CONFIG  — everything a reviewer needs to reproduce your numbers lives here
# ----------------------------------------------------------------------------------
# your registered Earth Engine project id
GCP_PROJECT = "map-project-413202"
# (minlon, minlat, maxlon, maxlat) ~Maiduguri MMC + Jere
AOI_BOUNDS = (13.05, 11.75, 13.30, 11.92)
# Sentinel-1 reference window (dry, wide enough to guarantee passes)
PRE_FLOOD = ("2024-08-01", "2024-09-08")
# tight PEAK window right after the 9-10 Sep dam collapse
DURING_FLOOD = ("2024-09-09", "2024-09-22")
# lock to ONE pass direction for geometric consistency.
ORBIT_PASS = "DESCENDING"
#   If a window comes back empty, flip to "ASCENDING" and rerun.
# during must be >= 3 dB DARKER than pre (a real backscatter drop)
DIFF_DB = -3.0
# during must be darker than this in absolute terms (open water is dark).
WATER_DB = -16.0
#   Tuning knob: raise toward -14 to catch more, lower toward -18 to be stricter.
# metres, focal-median smoothing to suppress SAR speckle
SPECKLE_RADIUS = 50
# drop flood polygons smaller than this (removes leftover specks)
MIN_AREA_M2 = 2000

PLACE = "Maiduguri, Borno, Nigeria"           # OSM geocode target

# Official reference: Copernicus EMS EMSR753 observed-flood polygon (authoritative extent).
OFFICIAL_FLOOD = r"D:\Users\SALWAABDULLA\Downloads\EMSR753_AOI01_GRA_MONIT01_v1\EMSR753_AOI01_GRA_MONIT01_observedEventA_v1.shp"
# mapping boundary, for the validation clip
OFFICIAL_AOI = OFFICIAL_FLOOD.replace("observedEventA", "areaOfInterestA")
# UTM 33N — metric CRS for area/overlap maths around Maiduguri
UTM_CRS = 32633


# ----------------------------------------------------------------------------------
# 1. SENTINEL-1 SAR FLOOD EXTENT (change detection)
# ----------------------------------------------------------------------------------
def sar_flood_image():
    """Compute the binary flood mask server-side and return (ee.Image, aoi).

    COPERNICUS/S1_GRD backscatter is already in DECIBELS (log scale), so change is a
    SUBTRACTION, not a ratio. We flag a pixel as flood only when BOTH hold:
      (a) during is >= |DIFF_DB| dB darker than pre  -> backscatter genuinely dropped
      (b) during is darker than WATER_DB in absolute terms -> it actually looks like open
          water (specular, dark). Condition (b) is what rejects bright buildings that
          merely 'changed', which is what smeared false positives across the whole city.
    Then remove permanent water (JRC) and steep terrain (HydroSHEDS) to isolate *new*
    flooding. Vectorization is done LOCALLY to stay under GEE quota.
    """
    ee.Initialize(project=GCP_PROJECT)
    aoi = ee.Geometry.Rectangle(list(AOI_BOUNDS))

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          # one look-geometry only
          .filter(ee.Filter.eq("orbitProperties_pass", ORBIT_PASS))
          .select("VV"))

    pre_col = s1.filterDate(*PRE_FLOOD)
    during_col = s1.filterDate(*DURING_FLOOD)

    n_pre, n_during = pre_col.size().getInfo(), during_col.size().getInfo()
    print(f"[SAR] {ORBIT_PASS} scenes:  pre={n_pre}  during={n_during}")
    if n_pre == 0 or n_during == 0:
        raise SystemExit(
            f"No {ORBIT_PASS} Sentinel-1 scenes in one window. Flip ORBIT_PASS to the "
            "other direction (or widen the dates) in CONFIG and rerun."
        )

    def smooth(img): return img.focal_median(
        SPECKLE_RADIUS, "circle", "meters")
    pre_db = smooth(pre_col.median())        # stable dry reference (median)
    # WETTEST pixel across the peak = max flood extent
    during_db = smooth(during_col.min())
    diff_db = during_db.subtract(pre_db)      # negative where it got darker

    flooded = diff_db.lte(DIFF_DB).And(
        during_db.lte(WATER_DB))  # dropped AND dark

    # Remove permanent water and steep slopes so we keep only *new* flooding.
    perm_water = ee.Image(
        "JRC/GSW1_4/GlobalSurfaceWater").select("seasonality").gte(5)
    flooded = flooded.where(perm_water, 0)
    slope = ee.Terrain.slope(ee.Image("WWF/HydroSHEDS/03VFDEM"))
    flooded = flooded.updateMask(slope.lt(5)).selfMask()

    return flooded.rename("flood").toByte(), aoi


def export_flood_geojson(path="flood_extent.geojson", tif="flood_mask.tif"):
    """Download a small flood-mask raster, vectorize LOCALLY, and drop speckle by area.

    Replaces reduceToVectors + ee_to_geojson (the quota-burning path). The raster is tiny
    at city scale, so GEE stays cheap even in restricted mode.
    """
    flooded, aoi = sar_flood_image()
    geemap.ee_export_image(flooded, filename=tif, scale=30,
                           region=aoi, file_per_band=False)
    print(f"[SAR] downloaded {tif}")

    with rasterio.open(tif) as src:
        band = src.read(1)
        polys = [
            {"geometry": geom, "properties": {"flood": 1}}
            for geom, val in rio_shapes(band, mask=(band == 1), transform=src.transform)
            if val == 1
        ]
        crs = src.crs
    gdf = gpd.GeoDataFrame.from_features(polys, crs=crs)

    # Minimum-mapping-unit filter: drop specks. Area needs a metric CRS (UTM 33N here).
    before = len(gdf)
    gdf = gdf.to_crs(32633)
    gdf = gdf[gdf.area >= MIN_AREA_M2].to_crs(4326)
    print(
        f"[SAR] polygons: {before} -> {len(gdf)} after dropping < {MIN_AREA_M2} m^2")

    gdf.to_file(path, driver="GeoJSON")
    print(f"[SAR] wrote {path}")
    return path


def export_vv_geotiffs(during_tif="during_vv.tif", pre_tif="pre_vv.tif"):
    """Export the pre + during VV backscatter (dB) composites as plain GeoTIFFs.

    These are ready-to-use test inputs for the FloodLens upload app: upload during_vv.tif
    (and pre_vv.tif to enable change detection) and it should reproduce this pipeline's
    Maiduguri result. Run this once with:  python maiduguri_flood_exposure.py --export-vv
    """
    ee.Initialize(project=GCP_PROJECT)
    aoi = ee.Geometry.Rectangle(list(AOI_BOUNDS))
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.eq("orbitProperties_pass", ORBIT_PASS))
          .select("VV"))

    def smooth(img): return img.focal_median(
        SPECKLE_RADIUS, "circle", "meters")
    during = smooth(s1.filterDate(*DURING_FLOOD).min()
                    ).rename("VV")    # wettest = peak
    pre = smooth(s1.filterDate(*PRE_FLOOD).median()
                 ).rename("VV")    # dry reference
    geemap.ee_export_image(during, filename=during_tif,
                           scale=30, region=aoi, file_per_band=False)
    geemap.ee_export_image(pre,    filename=pre_tif,
                           scale=30, region=aoi, file_per_band=False)
    print(
        f"[EXPORT] wrote {during_tif} and {pre_tif} — upload these to FloodLens to test.")
    return during_tif, pre_tif


# ----------------------------------------------------------------------------------
# 2. OSM INFRASTRUCTURE EXTRACTION  (documented, reproducible tags)
# ----------------------------------------------------------------------------------
OSM_TAGS = {
    "schools":   {"amenity": "school"},
    "clinics":   {"amenity": ["clinic", "hospital", "doctors"], "healthcare": True},
    "buildings": {"building": True},
}


def extract_osm(place=PLACE):
    """Pull each layer via OSMnx (Overpass under the hood). Roads come from the graph."""
    layers = {}
    for name, tags in OSM_TAGS.items():
        gdf = ox.features_from_place(place, tags)
        # normalise points/polys to representative points for clean spatial joins
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
        layers[name] = gdf.to_crs(4326)
        print(f"[OSM] {name}: {len(gdf)} features")

    roads = ox.graph_to_gdfs(ox.graph_from_place(
        place, network_type="drive"), nodes=False)
    layers["roads"] = roads.to_crs(4326)
    print(f"[OSM] roads: {len(roads)} segments")
    return layers


# ----------------------------------------------------------------------------------
# 3. OFFICIAL FLOOD LAYER + EXPOSURE OVERLAY
# ----------------------------------------------------------------------------------
def load_flood(path):
    """Load a flood polygon layer (shapefile or GeoJSON), repair geometry, to WGS84."""
    gdf = gpd.read_file(path).to_crs(4326)
    gdf["geometry"] = gdf.geometry.buffer(0)          # fix any invalid rings
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    return gdf


def compute_exposure(layers, flood_gdf, tag="official"):
    """Which OSM facilities fall inside the flood polygon. flood_gdf is a GeoDataFrame."""
    flood_union = flood_gdf.union_all()

    rows, exposed_frames = [], []
    for name, gdf in layers.items():
        pts = gdf.copy()
        # unambiguous point-in-polygon
        pts["geometry"] = pts.geometry.representative_point()
        hit = pts[pts.geometry.within(flood_union)]
        rows.append({"layer": name, "total": len(gdf), "exposed": len(hit)})
        if len(hit):
            h = hit.copy()
            h["layer"] = name
            exposed_frames.append(h[["layer", "geometry"]])

    summary = pd.DataFrame(rows)
    print(f"\n=== EXPOSURE SUMMARY (vs {tag} flood extent) ===")
    print(summary.to_string(index=False))
    summary.to_csv("exposure_summary.csv", index=False)

    if exposed_frames:
        gpd.GeoDataFrame(pd.concat(exposed_frames, ignore_index=True), crs=4326) \
            .to_file("exposed_facilities.geojson", driver="GeoJSON")
    return summary


# ----------------------------------------------------------------------------------
# 3b. VALIDATION — how well the independent SAR extent reproduces the official one
# ----------------------------------------------------------------------------------
def sar_vs_official(sar_path, official_gdf):
    """Report containment (fraction of SAR inside official) and IoU, clipped to the AOI.

    Containment is the honest headline: the official delineation is far larger (it includes
    hand-mapped urban inundation SAR can't see), so a raw IoU understates the agreement.
    Both are computed only where the two are comparable — inside the mapped AOI.
    """
    sar = load_flood(sar_path).to_crs(UTM_CRS)
    off = official_gdf.to_crs(UTM_CRS)

    try:                                                  # clip both to the official AOI box
        aoi = gpd.read_file(OFFICIAL_AOI).to_crs(UTM_CRS).union_all()
        sar_u = sar.union_all().intersection(aoi)
        off_u = off.union_all().intersection(aoi)
    except Exception:
        print("[VALIDATION] AOI layer not found — comparing without clipping.")
        sar_u, off_u = sar.union_all(), off.union_all()

    inter = sar_u.intersection(off_u).area
    containment = inter / sar_u.area if sar_u.area else 0.0
    iou = inter / (sar_u.area + off_u.area -
                   inter) if (sar_u.area + off_u.area - inter) else 0.0
    print("\n=== VALIDATION: independent Sentinel-1 SAR vs official EMSR753 ===")
    print(f"  SAR extent contained within official : {containment:5.1%}")
    print(f"  IoU (within mapped AOI)              : {iou:5.1%}")
    return containment, iou


# ----------------------------------------------------------------------------------
# 4. QUICK VALIDATION MAP  (sanity-check before you trust the numbers)
# ----------------------------------------------------------------------------------
def quick_map(layers, flood, out="exposure_map.html"):
    import folium
    minlon, minlat, maxlon, maxlat = AOI_BOUNDS
    m = folium.Map(location=[(minlat + maxlat) / 2,
                   (minlon + maxlon) / 2], zoom_start=12)
    folium.GeoJson(flood, name="flood", style_function=lambda _: {
                   "color": "#2b6cb0", "fillOpacity": 0.35}).add_to(m)
    for name in ("schools", "clinics"):
        for _, r in layers[name].iterrows():
            p = r.geometry.representative_point()
            folium.CircleMarker([p.y, p.x], radius=3, popup=name).add_to(m)
    folium.LayerControl().add_to(m)
    m.save(out)
    print(f"[MAP] wrote {out}")


if __name__ == "__main__":
    # Quick path: export VV GeoTIFFs to test the FloodLens app, then stop.
    if "--export-vv" in sys.argv:
        export_vv_geotiffs()
        sys.exit()

    # 1. Independent Sentinel-1 SAR extent — for validation + the floodplain story.
    sar_path = export_flood_geojson()

    # 2. OSM infrastructure.
    layers = extract_osm()

    # 3. Authoritative exposure numbers: overlay OSM on the official EMSR753 flood.
    official = load_flood(OFFICIAL_FLOOD)
    summary = compute_exposure(layers, official, tag="EMSR753 official")

    # 4. Quantify how well the independent SAR pipeline reproduces the official extent.
    sar_vs_official(sar_path, official)

    # 5. Sanity map (official flood + facilities).
    quick_map(layers, official)
    print("\nDone. exposure_summary.csv = headline numbers; "
          "exposed_facilities.geojson maps the exposed facilities.")
