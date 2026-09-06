"""Regenerate the offline cache-test wheel using only the Python standard library."""

import base64
import hashlib
from pathlib import Path
from zipfile import ZipFile, ZipInfo

dist_info = "cyberether_cache_fixture-1.0.dist-info"
files = {
    "cyberether_cache_fixture.py": "VALUE = 17\n",
    f"{dist_info}/METADATA": (
        "Metadata-Version: 2.1\nName: cyberether-cache-fixture\nVersion: 1.0\n"
    ),
    f"{dist_info}/WHEEL": (
        "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    ),
}
records = []
for name, text in files.items():
    data = text.encode()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    records.append(f"{name},sha256={digest},{len(data)}\n")
files[f"{dist_info}/RECORD"] = "".join(records) + f"{dist_info}/RECORD,,\n"

wheel = Path(__file__).with_name("cyberether_cache_fixture-1.0-py3-none-any.whl")
with ZipFile(wheel, "w") as archive:
    for name, text in files.items():
        info = ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
        info.external_attr = 0o644 << 16
        archive.writestr(info, text)
