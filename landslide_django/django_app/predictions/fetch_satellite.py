"""
predictions/fetch_satellite.py — FINAL WORKING VERSION

Strategy:
  1. If Copernicus credentials set → login, get token, download authenticated quicklook
  2. Always fallback to ESRI satellite tiles (no auth, instant, always works)
"""

import os
import math
import io
import requests
from pathlib import Path
from PIL import Image
from datetime import datetime, timedelta
from django.conf import settings

COPERNICUS_USER     = os.environ.get('COPERNICUS_USER', '').strip()
COPERNICUS_PASSWORD = os.environ.get('COPERNICUS_PASSWORD', '').strip()

SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL  = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


# ── Auth token ────────────────────────────────────────────────────────────────
def get_token():
    resp = requests.post(TOKEN_URL, data={
        "grant_type": "password",
        "client_id":  "cdse-public",
        "username":   COPERNICUS_USER,
        "password":   COPERNICUS_PASSWORD,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


# ── Search Copernicus ─────────────────────────────────────────────────────────
def search_sentinel2(lat, lon, days_back=90):
    delta      = 0.1
    end_date   = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)

    polygon = (
        f"{lon-delta} {lat-delta},{lon+delta} {lat-delta},"
        f"{lon+delta} {lat+delta},{lon-delta} {lat+delta},"
        f"{lon-delta} {lat-delta}"
    )
    params = {
        "$filter": (
            "Collection/Name eq 'SENTINEL-2' and "
            f"ContentDate/Start gt {start_date.strftime('%Y-%m-%dT00:00:00.000Z')} and "
            f"ContentDate/Start lt {end_date.strftime('%Y-%m-%dT23:59:59.000Z')} and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({polygon}))')"
        ),
        "$orderby": "ContentDate/Start desc",
        "$top": 5,
    }
    resp = requests.get(SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("value", [])
    if not results:
        raise ValueError("No Sentinel-2 scenes found for this location.")
    return results[0]["Id"], results[0]["Name"]


# ── Download quicklook with auth token ───────────────────────────────────────
def download_quicklook_authenticated(product_id, product_name, token, out_dir):
    headers    = {"Authorization": f"Bearer {token}"}
    save_dir   = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    urls = [
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/Assets(Quicklook)/$value",
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/Assets(QUICKLOOK)/$value",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 5000:
                out = save_dir / f"{product_name[:60]}_ql.jpg"
                out.write_bytes(r.content)
                return str(out)
        except Exception:
            continue
    raise ValueError("Authenticated quicklook download failed for all URL patterns.")


# ── ESRI satellite tile (no auth needed) ──────────────────────────────────────
def fetch_esri_stitched(lat, lon, zoom=14, save_dir=None):
    """
    Downloads a 3×3 grid of ESRI satellite tiles and stitches them.
    Works with zero authentication — always reliable.
    """

    save_dir = Path(save_dir or (Path(settings.MEDIA_ROOT) / "satellite_fetched"))
    save_dir.mkdir(parents=True, exist_ok=True)

    lat_r  = math.radians(lat)
    n      = 2 ** zoom
    cx     = int((lon + 180.0) / 360.0 * n)
    cy     = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)

    tile_px   = 256
    composite = Image.new("RGB", (tile_px * 3, tile_px * 3), (30, 30, 30))
    headers   = {"User-Agent": "LandslideAI/1.0"}
    fetched   = 0

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            url = (
                f"https://server.arcgisonline.com/ArcGIS/rest/services/"
                f"World_Imagery/MapServer/tile/{zoom}/{cy+dy}/{cx+dx}"
            )
            try:
                r = requests.get(url, headers=headers, timeout=20)
                if r.status_code == 200:
                    tile = Image.open(io.BytesIO(r.content)).convert("RGB")
                    composite.paste(tile, ((dx + 1) * tile_px, (dy + 1) * tile_px))
                    fetched += 1
            except Exception:
                pass

    if fetched == 0:
        raise ValueError("Could not download any ESRI tiles.")

    out = save_dir / f"satellite_{lat:.4f}_{lon:.4f}_z{zoom}.jpg"
    composite.save(str(out), "JPEG", quality=92)
    return str(out), f"Satellite imagery ({lat:.4f}°N, {lon:.4f}°E)"


# ── Main entry point ──────────────────────────────────────────────────────────
def fetch_satellite_image(lat, lon, save_dir=None):
    """
    Returns (image_path, scene_name).
    Tries Copernicus first, always falls back to ESRI tiles.
    """
    save_dir = Path(save_dir or (Path(settings.MEDIA_ROOT) / "satellite_fetched"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Try Copernicus if credentials are set ─────────────────────────────────
    if COPERNICUS_USER and COPERNICUS_PASSWORD:
        try:
            print("[satellite] Logging in to Copernicus...")
            token = get_token()
            product_id, product_name = search_sentinel2(lat, lon)
            img_path = download_quicklook_authenticated(
                product_id, product_name, token, save_dir
            )
            return img_path, f"Sentinel-2: {product_name[:50]}"
        except Exception as e:
            print(f"[satellite] Copernicus failed: {e} — falling back to ESRI tiles.")

    # ── Fallback: ESRI stitched tiles ─────────────────────────────────────────
    print("[satellite] Using ESRI satellite tiles...")
    return fetch_esri_stitched(lat, lon, zoom=14, save_dir=save_dir)