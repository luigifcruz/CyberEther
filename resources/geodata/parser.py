#!/bin/python3

import gzip
import json
import math
import os
import shutil
import struct
import sys
import urllib.request

import mapbox_earcut
import numpy as np


MAX_MERCATOR_LATITUDE = 85.05112878
PI_F32 = np.float32(math.pi)
DEGREES_TO_RADIANS_F32 = np.float32(math.pi / 180.0)


def format_hex_array(data, bytes_per_line=16):
    lines = []
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i : i + bytes_per_line]
        lines.append(", ".join("0x{:02x}".format(byte) for byte in chunk))
    return ",\n    ".join(lines)


def geographic_coordinate(coord):
    lon = float(np.float32(coord[0]))
    lat = float(np.float32(coord[1]))
    if lon < -180.0 or lon > 180.0:
        lon = math.fmod(lon + 180.0, 360.0)
        if lon < 0.0:
            lon += 360.0
        lon = float(np.float32(lon - 180.0))
    return lon, lat


def project_coordinate(coord):
    lon, lat = geographic_coordinate(coord)
    clamped_lat = np.float32(max(
        -MAX_MERCATOR_LATITUDE,
        min(MAX_MERCATOR_LATITUDE, lat),
    ))
    radians = np.float32(clamped_lat * DEGREES_TO_RADIANS_F32)
    tangent = np.float32(math.tan(float(radians)))
    mercator_x = np.float32(
        (np.float32(lon) + np.float32(180.0)) / np.float32(360.0)
    )
    mercator_y = np.float32(
        (
            np.float32(1.0)
            - np.float32(math.asinh(float(tangent))) / PI_F32
        )
        / np.float32(2.0)
    )
    mercator_y = np.clip(
        mercator_y,
        np.float32(0.0),
        np.float32(1.0),
    )
    return mercator_x, mercator_y


def triangulate_polygon(rings_coords):
    geographic_coords = []
    projected_coords = []
    ring_end_indices = []
    running_count = 0

    for ring in rings_coords:
        running_count += len(ring)
        ring_end_indices.append(running_count)
        for coord in ring:
            lon, lat = geographic_coordinate(coord)
            geographic_coords.extend((lon, lat))
            mercator_x, mercator_y = project_coordinate(coord)
            projected_coords.extend((mercator_x, mercator_y))

    if len(projected_coords) < 6:
        return [], [], []

    # Select topology in the projection where fills are currently rendered,
    # but retain geographic positions for other projections such as a globe.
    verts = np.array(projected_coords, dtype=np.float32).reshape(-1, 2)
    rings = np.array(ring_end_indices, dtype=np.uint32)

    indices = mapbox_earcut.triangulate_float32(verts, rings)

    return geographic_coords, verts.reshape(-1).tolist(), indices.tolist()


def process_geojson_triangulated(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_geographic_vertices = []
    all_projected_vertices = []
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
            geographic, projected, indices = triangulate_polygon(polygon)
            if not geographic or not projected or not indices:
                continue

            base_vertex = len(all_geographic_vertices) // 2
            all_geographic_vertices.extend(geographic)
            all_projected_vertices.extend(projected)
            all_indices.extend(idx + base_vertex for idx in indices)

    return all_geographic_vertices, all_projected_vertices, all_indices


def append_line_segment(start, end, geographic_vertices, projected_vertices):
    geographic_vertices.extend(start)
    geographic_vertices.extend(end)
    projected_vertices.extend(project_coordinate(start))
    projected_vertices.extend(project_coordinate(end))


def append_line_string(coords, geographic_vertices, projected_vertices):
    for index in range(len(coords) - 1):
        start = geographic_coordinate(coords[index])
        end = geographic_coordinate(coords[index + 1])
        delta = end[0] - start[0]
        if abs(delta) <= 180.0:
            append_line_segment(
                start, end, geographic_vertices, projected_vertices
            )
            continue

        unwrapped_end_lon = end[0] + (360.0 if delta < 0.0 else -360.0)
        unwrapped_delta = unwrapped_end_lon - start[0]
        if abs(unwrapped_delta) < 1e-7:
            append_line_segment(
                start, end, geographic_vertices, projected_vertices
            )
            continue

        boundary_lon = 180.0 if unwrapped_end_lon > 180.0 else -180.0
        fraction = (boundary_lon - start[0]) / unwrapped_delta
        boundary_lat = start[1] + (end[1] - start[1]) * fraction
        opposite_lon = -boundary_lon

        append_line_segment(
            start,
            (boundary_lon, boundary_lat),
            geographic_vertices,
            projected_vertices,
        )
        append_line_segment(
            (opposite_lon, boundary_lat),
            end,
            geographic_vertices,
            projected_vertices,
        )


def append_line_geometry(geometry, geographic_vertices, projected_vertices):
    if not geometry or not geometry.get("coordinates"):
        return

    geometry_type = geometry.get("type", "")
    coords = geometry["coordinates"]
    if geometry_type == "LineString":
        append_line_string(coords, geographic_vertices, projected_vertices)
    elif geometry_type == "MultiLineString":
        for line in coords:
            append_line_string(line, geographic_vertices, projected_vertices)
    elif geometry_type == "Polygon":
        for ring in coords:
            append_line_string(ring, geographic_vertices, projected_vertices)
    elif geometry_type == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                append_line_string(
                    ring, geographic_vertices, projected_vertices
                )


def process_geojson_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    geographic_vertices = []
    projected_vertices = []
    features = data.get("features", [])
    if not features:
        features = [{"geometry": data}]

    for feature in features:
        append_line_geometry(
            feature.get("geometry"),
            geographic_vertices,
            projected_vertices,
        )

    return geographic_vertices, projected_vertices


def emit_triangulated(fh, filename, filepath):
    print(f"  Pre-triangulating {filename}...")
    geographic, projected, indices = process_geojson_triangulated(filepath)
    if len(geographic) != len(projected):
        raise ValueError("Triangulated coordinate streams are misaligned")

    vertex_count = len(geographic) // 2
    index_count = len(indices)

    print(
        f"    {vertex_count} vertices, "
        f"{index_count} indices "
        f"({index_count // 3} triangles)"
    )

    binary = struct.pack("<II", vertex_count, index_count)
    if geographic:
        binary += struct.pack(f"<{len(geographic)}f", *geographic)
        binary += struct.pack(f"<{len(projected)}f", *projected)
    if indices:
        binary += struct.pack(f"<{index_count}I", *indices)

    compressed = gzip.compress(binary, compresslevel=9)
    size = len(compressed)
    raw_size = len(binary)

    hex_data = format_hex_array(compressed)

    fh.write(f"static const uint8_t {filename}_tri_gz[] = {{\n    {hex_data}\n}};\n")
    fh.write(f"static const uint32_t {filename}_tri_gz_len = {size};\n")
    fh.write(f"static const uint32_t {filename}_tri_raw_len = {raw_size};\n\n")


def emit_line_segments(fh, filename, filepath):
    print(f"  Packing line segments for {filename}...")
    geographic, projected = process_geojson_lines(filepath)
    if len(geographic) != len(projected):
        raise ValueError("Line coordinate streams are misaligned")
    print(f"    {len(geographic) // 4} line segments")

    binary = struct.pack("<I", len(geographic))
    if geographic:
        binary += struct.pack(f"<{len(geographic)}f", *geographic)
        binary += struct.pack(f"<{len(projected)}f", *projected)

    compressed = gzip.compress(binary, compresslevel=9)
    hex_data = format_hex_array(compressed)
    fh.write(
        f"static const uint8_t {filename}_segments_gz[] = "
        f"{{\n    {hex_data}\n}};\n"
    )
    fh.write(
        f"static const uint32_t {filename}_segments_gz_len = "
        f"{len(compressed)};\n"
    )
    fh.write(
        f"static const uint32_t {filename}_segments_raw_len = "
        f"{len(binary)};\n\n"
    )


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

GEODATA_BASE_URL = "https://cdn.cyberether.org/geodata/"
GEODATA_URL_OVERRIDES = {
    "ne_10m_geographic_lines.geojson": (
        "https://raw.githubusercontent.com/nvkelso/"
        "natural-earth-vector/v5.1.2/geojson/"
        "ne_10m_geographic_lines.geojson"
    ),
}


def download_if_missing(filepath):
    if os.path.exists(filepath):
        return
    filename = os.path.basename(filepath)
    url = GEODATA_URL_OVERRIDES.get(filename, GEODATA_BASE_URL + filename)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CyberEther/1.0 (+https://github.com/luigifcruz/CyberEther)"},
    )
    temporary_filepath = filepath + ".part"
    print(f"[GEODATA] Downloading {filename}...")
    try:
        with urllib.request.urlopen(request) as response:
            with open(temporary_filepath, "wb") as output:
                shutil.copyfileobj(response, output)
        os.replace(temporary_filepath, filepath)
    except Exception as e:
        if os.path.exists(temporary_filepath):
            os.remove(temporary_filepath)
        print(f"[GEODATA] Failed to download {filename}: {e}", file=sys.stderr)
        sys.exit(1)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[GEODATA] Done ({size_mb:.1f} MB).")


# Usage: parser.py <build_root> <output> [options] <inputs...>
# Files whose basename (without extension) matches a name in the
# --triangulate list are pre-triangulated into binary vertex/index
# buffers. Files in --line-segments are packed into binary line segments.
# All other files are gzip-compressed as-is.

if __name__ == "__main__":
    path = sys.argv[1]
    output = sys.argv[2]

    tri_set = set()
    line_set = set()
    inputs = []

    for arg in sys.argv[3:]:
        if arg.startswith("--triangulate="):
            names = arg[len("--triangulate=") :]
            tri_set.update(n.strip() for n in names.split(","))
        elif arg.startswith("--line-segments="):
            names = arg[len("--line-segments=") :]
            line_set.update(n.strip() for n in names.split(","))
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

    output_path = os.path.join(path, output)
    if not os.path.isdir(os.path.dirname(output_path)):
        output_path = output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as fh:
        fh.write("#pragma once\n\n")
        fh.write("#include <stdint.h>\n\n")
        fh.write("namespace Jetstream::Resources {\n\n")

        for filepath in inputs:
            filename = os.path.basename(filepath).split(".")[0]

            if filename in tri_set:
                emit_triangulated(fh, filename, filepath)
            elif filename in line_set:
                emit_line_segments(fh, filename, filepath)
            else:
                emit_compressed(fh, filename, filepath)

        fh.write("\n}  // namespace Jetstream::Resources\n")
