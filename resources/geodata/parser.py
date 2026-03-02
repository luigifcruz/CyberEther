#!/bin/python3

import gzip
import json
import os
import struct
import sys
import urllib.request

import mapbox_earcut
import numpy as np


def format_hex_array(data, bytes_per_line=16):
    lines = []
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i : i + bytes_per_line]
        lines.append(", ".join("0x{:02x}".format(byte) for byte in chunk))
    return ",\n    ".join(lines)


def triangulate_polygon(rings_coords):
    flat_coords = []
    ring_end_indices = []
    running_count = 0

    for ring in rings_coords:
        running_count += len(ring)
        ring_end_indices.append(running_count)
        for coord in ring:
            flat_coords.append(coord[0])
            flat_coords.append(coord[1])

    if len(flat_coords) < 6:
        return [], []

    verts = np.array(flat_coords, dtype=np.float32).reshape(-1, 2)
    rings = np.array(ring_end_indices, dtype=np.uint32)

    indices = mapbox_earcut.triangulate_float32(verts, rings)

    return flat_coords, indices.tolist()


def process_geojson_triangulated(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_vertices = []
    all_indices = []

    features = data.get("features", [])
    if not features:
        features = [{"geometry": data}]

    for feature in features:
        geom = feature.get("geometry")
        if not geom:
            continue

        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        polygons = []
        if gtype == "Polygon":
            polygons = [coords]
        elif gtype == "MultiPolygon":
            polygons = coords

        for polygon in polygons:
            verts, indices = triangulate_polygon(polygon)
            if not verts or not indices:
                continue

            base_vertex = len(all_vertices) // 2
            all_vertices.extend(verts)
            all_indices.extend(idx + base_vertex for idx in indices)

    return all_vertices, all_indices


def emit_triangulated(fh, filename, filepath):
    print(f"  Pre-triangulating {filename}...")
    vertices, indices = process_geojson_triangulated(filepath)

    vertex_count = len(vertices) // 2
    index_count = len(indices)

    print(
        f"    {vertex_count} vertices, "
        f"{index_count} indices "
        f"({index_count // 3} triangles)"
    )

    binary = struct.pack("<II", vertex_count, index_count)
    if vertices:
        binary += struct.pack(f"<{len(vertices)}f", *vertices)
    if indices:
        binary += struct.pack(f"<{index_count}I", *indices)

    compressed = gzip.compress(binary, compresslevel=9)
    size = len(compressed)
    raw_size = len(binary)

    hex_data = format_hex_array(compressed)

    fh.write(f"static const uint8_t {filename}_tri_gz[] = {{\n    {hex_data}\n}};\n")
    fh.write(f"static const uint32_t {filename}_tri_gz_len = {size};\n")
    fh.write(f"static const uint32_t {filename}_tri_raw_len = {raw_size};\n\n")


def emit_compressed(fh, filename, filepath):
    with open(filepath, "rb") as f:
        raw = f.read()

    compressed = gzip.compress(raw, compresslevel=9)
    size = len(compressed)
    raw_size = len(raw)

    hex_data = format_hex_array(compressed)

    fh.write(f"static const uint8_t {filename}_gz[] = {{\n    {hex_data}\n}};\n")
    fh.write(f"static const uint32_t {filename}_gz_len = {size};\n")
    fh.write(f"static const uint32_t {filename}_raw_len = {raw_size};\n\n")


# Download missing GeoJSON files from Natural Earth.

GEODATA_BASE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
)


def download_if_missing(filepath):
    if os.path.exists(filepath):
        return
    filename = os.path.basename(filepath)
    url = GEODATA_BASE_URL + filename
    print(f"[GEODATA] Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filepath)
    except Exception as e:
        print(f"[GEODATA] Failed to download {filename}: {e}", file=sys.stderr)
        sys.exit(1)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[GEODATA] Done ({size_mb:.1f} MB).")


# Usage: parser.py <build_root> <output> [--triangulate=a,b,...] <inputs...>
# Files whose basename (without extension) matches a name in the
# --triangulate list are pre-triangulated into binary vertex/index
# buffers. All other files are gzip-compressed as-is.

if __name__ == "__main__":
    path = sys.argv[1]
    output = sys.argv[2]

    tri_set = set()
    inputs = []

    for arg in sys.argv[3:]:
        if arg.startswith("--triangulate="):
            names = arg[len("--triangulate=") :]
            tri_set.update(n.strip() for n in names.split(","))
        else:
            inputs.append(arg)

    # Download any missing GeoJSON files.
    missing = [f for f in inputs if not os.path.exists(f)]
    if missing:
        print(
            f"[GEODATA] Downloading {len(missing)} missing "
            f"GeoJSON file(s) from Natural Earth..."
        )
        for filepath in missing:
            download_if_missing(filepath)
        print(f"[GEODATA] All downloads complete.")

    print(f"[GEODATA] Compiling {len(inputs)} geodata files...")

    with open(os.path.join(path, output), "w") as fh:
        fh.write("#pragma once\n\n")
        fh.write("#include <stdint.h>\n\n")
        fh.write("namespace Jetstream::Resources {\n\n")

        for filepath in inputs:
            filename = os.path.basename(filepath).split(".")[0]

            if filename in tri_set:
                emit_triangulated(fh, filename, filepath)
            else:
                emit_compressed(fh, filename, filepath)

        fh.write("\n}  // namespace Jetstream::Resources\n")
