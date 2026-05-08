import requests, math
from pathlib import Path

# ── Put your Copernicus credentials here ──────────────────────────────────────
COPERNICUS_USER     = "lybu369@gmail.com"   # ← change this
COPERNICUS_PASSWORD = "MasterLee@369"             # ← change this
# Register free at: https://dataspace.copernicus.eu

lat, lon    = 13.222834, 76.277590
PRODUCT_ID  = "3d78a4c9-3dea-497e-8a5a-7a358f2099c8"   # from previous result
PRODUCT_NAME= "S2B_MSIL1C_20260505T051649"

# ── Step 1: Get access token ──────────────────────────────────────────────────
print("Step 1 — Getting access token...")
token_resp = requests.post(
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    data={
        "grant_type": "password",
        "client_id":  "cdse-public",
        "username":   COPERNICUS_USER,
        "password":   COPERNICUS_PASSWORD,
    },
    timeout=30
)
print(f"Token status: {token_resp.status_code}")

if token_resp.status_code != 200:
    print(f"Login failed: {token_resp.text[:300]}")
    print("\nFalling back to ESRI tile...")
    # Fallback to ESRI tile
    zoom   = 14
    lat_r  = __import__('math').radians(lat)
    n      = 2 ** zoom
    xt     = int((lon + 180.0) / 360.0 * n)
    yt     = int((1.0 - __import__('math').log(__import__('math').tan(lat_r) + 1.0 / __import__('math').cos(lat_r)) / __import__('math').pi) / 2.0 * n)
    r2     = requests.get(f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{yt}/{xt}", headers={"User-Agent":"LandslideAI/1.0"}, timeout=30)
    Path("fallback_tile.jpg").write_bytes(r2.content)
    print("Saved fallback → fallback_tile.jpg")
    exit()

token = token_resp.json()["access_token"]
print("Token obtained successfully!")

headers = {"Authorization": f"Bearer {token}"}

# ── Step 2: Try quicklook URL patterns with auth ──────────────────────────────
print("\nStep 2 — Trying authenticated quicklook URLs...")

urls_to_try = [
    f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({PRODUCT_ID})/Assets(Quicklook)/$value",
    f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({PRODUCT_ID})/Assets(QUICKLOOK)/$value",
    f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({PRODUCT_ID})/$value",
]

for url in urls_to_try:
    print(f"Trying: {url[:90]}...")
    r = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    print(f"  Status: {r.status_code} | Size: {len(r.content)} bytes | Type: {r.headers.get('Content-Type','?')}")
    if r.status_code == 200 and len(r.content) > 5000:
        out = Path("sentinel2_authenticated.jpg")
        out.write_bytes(r.content)
        print(f"  Saved → {out.resolve()}")
        break
    else:
        print(f"  Response: {r.text[:100]}")