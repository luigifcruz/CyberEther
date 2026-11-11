#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(manifest_path, output, cache_dir):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["format_version"] != 3:
        raise ValueError("Unsupported geodata format; expected version 3.")
    for key in ("sha256",):
        value = manifest[key]
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid {key} in geodata manifest.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / (manifest["sha256"] + ".bin")
    if not archive.is_file() or sha256(archive) != manifest["sha256"]:
        print(f"[GEODATA] Downloading precompiled maps from {manifest['url']}...")
        # A checksum-qualified request bypasses stale CDN misses from publication.
        url = manifest["url"] + "?sha256=" + manifest["sha256"]
        request = urllib.request.Request(url, headers={
            "User-Agent": "CyberEther-Build",
            "Accept-Encoding": "identity",
        })
        # Unique temporary paths also support concurrent native/browser builds.
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".part", delete=False) as target:
            temporary = Path(target.name)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with temporary.open("wb") as target:
                    shutil.copyfileobj(response, target)
            if sha256(temporary) != manifest["sha256"]:
                raise ValueError("Downloaded geodata failed SHA-256 verification.")
            os.replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".part", delete=False) as target:
        temporary = Path(target.name)
    try:
        # Use the same generated C++ byte-array pattern as resources/remote.
        with archive.open("rb") as source, temporary.open("w", encoding="utf-8") as target:
            target.write("#include <cstddef>\n\n")
            target.write("namespace Jetstream::Resources {\n\n")
            target.write("extern const unsigned char GeoDataBundle[] = {\n")
            for chunk in iter(lambda: source.read(16), b""):
                target.write("    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",\n")
            target.write("};\n\n")
            target.write(f"extern const std::size_t GeoDataBundleSize = {archive.stat().st_size};\n\n")
            target.write("}  // namespace Jetstream::Resources\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Fetch precompiled CyberEther map data.")
    cli.add_argument("manifest", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--cache-dir", type=Path, required=True)
    args = cli.parse_args()
    try:
        fetch(args.manifest, args.output, args.cache_dir)
    except Exception as error:
        sys.exit(f"[GEODATA] Failed to fetch precompiled maps: {error}")
