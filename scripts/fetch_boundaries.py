"""Download the 31 sub-district (kecamatan) boundaries of Surabaya used in Fig. 2.

The boundaries are GADM 4.1 level-3 polygons. GADM may be used freely for academic and
other non-commercial work but may not be redistributed, so this repository does not ship
them. This script takes them from the npm package geojson-indonesia 1.0.1, which bundles
the GADM 4.1 files for Indonesia, checks the archive hash, and writes only the Surabaya
features to data/boundaries/surabaya_kecamatan.geojson.

Source: https://gadm.org (GADM 4.1); https://www.npmjs.com/package/geojson-indonesia
"""
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://registry.npmjs.org/geojson-indonesia/-/geojson-indonesia-1.0.1.tgz"
SHA1 = "53d5c6f4f1ea1b57f8b23e6073347d5aa7a47532"
MEMBER = "package/geojson/gadm41_IDN_3.json"
OUT = ROOT / "data" / "boundaries" / "surabaya_kecamatan.geojson"


def main():
    if OUT.exists():
        print(f"{OUT.relative_to(ROOT)} already exists")
        return
    print(f"downloading {URL} (about 13 MB)")
    with urllib.request.urlopen(URL, timeout=120) as r:
        blob = r.read()
    digest = hashlib.sha1(blob).hexdigest()
    if digest != SHA1:
        raise SystemExit(f"hash mismatch: got {digest}, expected {SHA1}")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        level3 = json.load(tar.extractfile(MEMBER))
    feats = [f for f in level3["features"] if f["properties"].get("NAME_2") == "Surabaya"]
    if len(feats) != 31:
        raise SystemExit(f"expected 31 Surabaya sub-districts, found {len(feats)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"type": "FeatureCollection",
               "source": "GADM 4.1 (https://gadm.org), via geojson-indonesia 1.0.1; not for redistribution",
               "features": feats}, open(OUT, "w"))
    print(f"wrote {OUT.relative_to(ROOT)} ({len(feats)} sub-districts)")


if __name__ == "__main__":
    main()
