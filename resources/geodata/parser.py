#!/bin/python3

import gzip
import heapq
import json
import math
import os
import shutil
import struct
import sys
import unicodedata
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


def project_unwrapped_coordinate(lon, lat):
    # Mercator projection without longitude wrapping: longitudes outside
    # [-180, 180] map to x outside [0, 1], keeping antimeridian-crossing
    # rings contiguous in the triangulation space.
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
    # Unwrap each ring's longitudes around a per-polygon reference so rings
    # crossing the antimeridian stay contiguous in the triangulation space.
    # Triangulating in wrapped Mercator space bridges the +/-180 seam with
    # giant triangles whose 3D chords cut through the globe interior,
    # leaving uncovered holes (and floating shards) on the sphere.
    unwrapped_rings = []
    reference_lon = None
    for ring in rings_coords:
        if not ring:
            continue
        unwrapped = unwrap_ring(ring, reference_lon)
        if reference_lon is None:
            reference_lon = (
                sum(coord[0] for coord in unwrapped) / len(unwrapped)
            )
        unwrapped_rings.append(unwrapped)

    geographic_coords = []
    projected_coords = []
    ring_end_indices = []
    running_count = 0

    for ring in unwrapped_rings:
        running_count += len(ring)
        ring_end_indices.append(running_count)
        for lon, lat in ring:
            geographic_coords.extend((lon, lat))
            mercator_x, mercator_y = project_unwrapped_coordinate(lon, lat)
            projected_coords.extend((mercator_x, mercator_y))

    if len(projected_coords) < 6:
        return [], []

    # Triangulate in the unwrapped Mercator plane for earcut topology; the
    # geographic stream is what the globe renderer projects onto the sphere.
    verts = np.array(projected_coords, dtype=np.float32).reshape(-1, 2)
    rings = np.array(ring_end_indices, dtype=np.uint32)

    indices = mapbox_earcut.triangulate_float32(verts, rings).tolist()

    # Subdivide triangles whose edges span too much of the sphere: fills are
    # rendered with straight 3D chords on the unit globe, so a large flat
    # triangle dips inside the surface by 1-cos(arc/2) and opens crescent
    # holes near the limb where the starfield shows through the globe.
    geographic_coords, indices = subdivide_triangles_for_sphere(
        geographic_coords, indices
    )

    # Stored coordinates are wrapped back into range for the binary format;
    # the sphere mapping resolves the antimeridian natively.
    for index in range(0, len(geographic_coords), 2):
        lon, lat = geographic_coordinate(
            (geographic_coords[index], geographic_coords[index + 1])
        )
        geographic_coords[index] = lon
        geographic_coords[index + 1] = lat
    return geographic_coords, indices


# Maximum great-circle span (degrees) of a single fill-triangle edge on the
# globe. At 6 degrees the chord dips only ~1.4e-3 of the globe radius below
# the surface (~1px at zoom 0, the only zoom range where the silhouette is
# visible); the procedural water sphere underpaints any remaining sliver.
MAX_FILL_EDGE_DEGREES = 6.0


def _sphere_unit(lon, lat):
    r = math.radians(lat)
    lr = math.radians(lon)
    cl = math.cos(r)
    return (cl * math.sin(lr), math.sin(r), cl * math.cos(lr))


def _sphere_arc_degrees(a, b):
    cross = (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
    cross_norm = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return math.degrees(math.atan2(cross_norm, dot))


def _wrap_longitude_near(lon, reference):
    while lon - reference > 180.0:
        lon -= 360.0
    while lon - reference < -180.0:
        lon += 360.0
    return lon


def subdivide_triangles_for_sphere(geographic, indices):
    """Split fill triangles until every edge is short on the sphere.

    Edge midpoints are shared through a cache keyed by the endpoint pair so
    adjacent triangles stay watertight (no T-junction cracks). Midpoints lie
    on the unit sphere, keeping the mesh spherical after subdivision.
    Operates on unwrapped longitudes; storage wrapping happens afterwards.
    """
    max_arc = MAX_FILL_EDGE_DEGREES
    points = [
        _sphere_unit(geographic[i], geographic[i + 1])
        for i in range(0, len(geographic), 2)
    ]
    geographic = list(geographic)
    out_indices = []
    midpoint_cache = {}

    def midpoint(index_a, index_b):
        key = (index_a, index_b) if index_a < index_b else (index_b, index_a)
        cached = midpoint_cache.get(key)
        if cached is not None:
            return cached
        a = points[index_a]
        b = points[index_b]
        mx = a[0] + b[0]
        my = a[1] + b[1]
        mz = a[2] + b[2]
        norm = math.sqrt(mx * mx + my * my + mz * mz)
        if norm < 1e-12:
            # Antipodal endpoints: pick any perpendicular direction.
            mx, my, mz = -a[1], a[0], 0.0
            norm = math.sqrt(mx * mx + my * my + mz * mz)
            if norm < 1e-12:
                mx, my, mz = 0.0, -a[2], a[1]
                norm = math.sqrt(mx * mx + my * my + mz * mz)
        mx, my, mz = mx / norm, my / norm, mz / norm
        lon = math.degrees(math.atan2(mx, mz))
        lat = math.degrees(math.asin(max(-1.0, min(1.0, my))))
        # Keep the new vertex in the same longitude copy as its edge.
        reference = 0.5 * (
            geographic[2 * index_a] + geographic[2 * index_b]
        )
        lon = _wrap_longitude_near(lon, reference)
        points.append((mx, my, mz))
        geographic.extend((lon, lat))
        new_index = len(points) - 1
        midpoint_cache[key] = new_index
        return new_index

    stack = [
        (indices[i], indices[i + 1], indices[i + 2])
        for i in range(0, len(indices), 3)
    ]
    while stack:
        a, b, c = stack.pop()
        arc_ab = _sphere_arc_degrees(points[a], points[b])
        arc_bc = _sphere_arc_degrees(points[b], points[c])
        arc_ca = _sphere_arc_degrees(points[c], points[a])
        longest = max(arc_ab, arc_bc, arc_ca)
        if longest <= max_arc:
            out_indices.extend((a, b, c))
            continue
        if longest == arc_ab:
            m = midpoint(a, b)
            stack.append((a, m, c))
            stack.append((m, b, c))
        elif longest == arc_bc:
            m = midpoint(b, c)
            stack.append((b, m, a))
            stack.append((m, c, a))
        else:
            m = midpoint(c, a)
            stack.append((c, m, b))
            stack.append((m, a, b))

    return geographic, out_indices


def process_geojson_triangulated(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_geographic_vertices = []
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
            geographic, indices = triangulate_polygon(polygon)
            if not geographic or not indices:
                continue

            base_vertex = len(all_geographic_vertices) // 2
            all_geographic_vertices.extend(geographic)
            all_indices.extend(idx + base_vertex for idx in indices)

    return all_geographic_vertices, all_indices


def append_line_segment(start, end, geographic_vertices):
    geographic_vertices.extend(start)
    geographic_vertices.extend(end)


def append_line_string(coords, geographic_vertices):
    for index in range(len(coords) - 1):
        start = geographic_coordinate(coords[index])
        end = geographic_coordinate(coords[index + 1])
        delta = end[0] - start[0]
        if abs(delta) <= 180.0:
            append_line_segment(
                start, end, geographic_vertices
            )
            continue

        unwrapped_end_lon = end[0] + (360.0 if delta < 0.0 else -360.0)
        unwrapped_delta = unwrapped_end_lon - start[0]
        if abs(unwrapped_delta) < 1e-7:
            append_line_segment(
                start, end, geographic_vertices
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
        )
        append_line_segment(
            (opposite_lon, boundary_lat),
            end,
            geographic_vertices,
        )


def append_line_geometry(geometry, geographic_vertices):
    if not geometry or not geometry.get("coordinates"):
        return

    geometry_type = geometry.get("type", "")
    coords = geometry["coordinates"]
    if geometry_type == "LineString":
        append_line_string(coords, geographic_vertices)
    elif geometry_type == "MultiLineString":
        for line in coords:
            append_line_string(line, geographic_vertices)
    elif geometry_type == "Polygon":
        for ring in coords:
            append_line_string(ring, geographic_vertices)
    elif geometry_type == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                append_line_string(
                    ring, geographic_vertices
                )


def process_geojson_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    geographic_vertices = []
    features = data.get("features", [])
    if not features:
        features = [{"geometry": data}]

    for feature in features:
        append_line_geometry(
            feature.get("geometry"),
            geographic_vertices,
        )

    return geographic_vertices


def emit_blob(fh, symbol, binary):
    """Emit one deterministic gzip resource and its compressed/raw lengths."""
    compressed = gzip.compress(binary, compresslevel=9, mtime=0)
    fh.write(
        f"static const uint8_t {symbol}_gz[] = "
        f"{{\n    {format_hex_array(compressed)}\n}};\n"
    )
    fh.write(f"static const uint32_t {symbol}_gz_len = {len(compressed)};\n")
    fh.write(f"static const uint32_t {symbol}_raw_len = {len(binary)};\n\n")


def emit_triangulated(fh, filename, filepath):
    print(f"  Pre-triangulating {filename}...")
    geographic, indices = process_geojson_triangulated(filepath)

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
    if indices:
        binary += struct.pack(f"<{index_count}I", *indices)

    emit_blob(fh, filename + "_tri", binary)


def emit_line_segments(fh, filename, filepath):
    print(f"  Packing line segments for {filename}...")
    geographic = process_geojson_lines(filepath)
    print(f"    {len(geographic) // 4} line segments")

    binary = struct.pack("<I", len(geographic))
    if geographic:
        binary += struct.pack(f"<{len(geographic)}f", *geographic)
    emit_blob(fh, filename + "_segments", binary)


DISPUTED_LINE_CLASSES = {
    "Breakaway": 0,
    "Claim boundary": 1,
    "Elusive frontier": 2,
    "Reference line": 3,
}


def river_line_category(props):
    try:
        raw_rank = props.get("scalerank")
        rank = int(round(float(10 if raw_rank is None else raw_rank)))
    except (TypeError, ValueError):
        rank = 10
    return max(0, min(rank, 10))


def emit_styled_line_segments(fh, filename, filepath):
    """Pack categorized lines while retaining class and minimum zoom."""
    print(f"  Packing styled line segments for {filename}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    geographic = []
    min_zooms = []
    classes = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        if filename == "ne_10m_rivers_lake_centerlines":
            category = river_line_category(props)
            min_zoom = float(props.get("min_zoom") or 0.0)
        else:
            feature_class = props.get("FEATURECLA")
            if feature_class not in DISPUTED_LINE_CLASSES:
                continue
            category = DISPUTED_LINE_CLASSES[feature_class]
            min_zoom = float(props.get("MIN_ZOOM") or 0.0)
        before = len(geographic) // 4
        append_line_geometry(feature.get("geometry"), geographic)
        added = len(geographic) // 4 - before
        min_zooms.extend([min_zoom] * added)
        classes.extend([category] * added)

    segment_count = len(geographic) // 4
    if len(min_zooms) != segment_count:
        raise ValueError("Styled line streams are misaligned")

    binary = struct.pack("<I", segment_count)
    if segment_count:
        binary += struct.pack(f"<{len(geographic)}f", *geographic)
        binary += struct.pack(f"<{segment_count}f", *min_zooms)
        binary += struct.pack(f"<{segment_count}B", *classes)

    emit_blob(fh, filename + "_styled_segments", binary)
    print(f"    {segment_count} styled line segments")


# Split ne_10m_geographic_lines into four styling categories so the renderer
# can apply a distinct style to each (equator solid, tropics/polar dashed,
# International Date Line solid and thick). Emits one segment blob per group.
GEOGRAPHIC_LINE_GROUPS = {
    "equator": {"Equator"},
    "tropics": {"Tropic of Cancer", "Tropic of Capricorn"},
    "polar": {"Arctic Circle", "Antarctic Circle"},
    "dateline": {"International Date Line"},
}


def emit_geographic_lines_split(fh, filename, filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    by_group = {g: [] for g in GEOGRAPHIC_LINE_GROUPS}
    for feature in features:
        name = (feature.get("properties") or {}).get("name")
        for group, names in GEOGRAPHIC_LINE_GROUPS.items():
            if name in names:
                by_group[group].append(feature)
                break

    for group, feats in by_group.items():
        print(f"  Packing geographic line group '{group}' "
              f"({len(feats)} feature(s))...")
        geographic = []
        for feature in feats:
            append_line_geometry(feature.get("geometry"), geographic)

        sym = f"{filename}_{group}"
        binary = struct.pack("<I", len(geographic))
        if geographic:
            binary += struct.pack(f"<{len(geographic)}f", *geographic)
        emit_blob(fh, sym + "_segments", binary)
        print(f"    {len(geographic) // 4} line segments")


# Label-point binary emitter.
#
# Binary format: [U32 record_count]
#                per record:
#                  [F32 lon, lat]
#                  [I32 scalerank][F32 minZoom][F32 maxZoom][I32 population]
#                  [F32 area]  (latitude-corrected square degrees, states only)
#                  [U8 kind][U8 flags][U8 nameLen][U8 pathPointCount]
#                  [char[nameLen] name]   (ASCII 32-126)
#                  [F32 lon, lat * pathPointCount]
#
# kind: 0=City,1=Capital,2=Country,3=State,4=Water,5=Physical,
#       6=Airport,7=River,8=Lake,9=Region
# flags: bit0 capital, bit1 worldcity, bit2 megacity, bit3 national capital

KIND_CITY = 0
KIND_CAPITAL = 1
KIND_COUNTRY = 2
KIND_STATE = 3
KIND_WATER = 4
KIND_PHYSICAL = 5
KIND_AIRPORT = 6
KIND_RIVER = 7
KIND_LAKE = 8
KIND_REGION = 9

MAX_LABEL_NAME = 255  # The binary record stores the ASCII byte length in a U8.
MAX_LABEL_PATH_POINTS = 9

MARINE_LABEL_OVERRIDES = {
    "Arctic Ocean": (-150.0, 78.0),
    "North Atlantic Ocean": (-35.0, 35.0),
    "South Atlantic Ocean": (-15.0, -35.0),
    "North Pacific Ocean": (-160.0, 30.0),
    "South Pacific Ocean": (-130.0, -35.0),
    "INDIAN OCEAN": (80.0, -25.0),
    "SOUTHERN OCEAN": (0.0, -60.0),
}


ASCII_REPLACEMENTS = str.maketrans({
    "Æ": "AE", "æ": "ae", "Œ": "OE", "œ": "oe",
    "Ø": "O", "ø": "o", "Ł": "L", "ł": "l",
    "Đ": "D", "đ": "d", "Ð": "D", "ð": "d",
    "Þ": "Th", "þ": "th", "ß": "ss",
    "Ə": "E", "ə": "e", "Ħ": "H", "ħ": "h",
    "ı": "i", "Ŋ": "N", "ŋ": "n", "Ŧ": "T", "ŧ": "t",
    "ʻ": "'", "ʼ": "'",
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
})


def transliterate_ascii(value):
    value = value.translate(ASCII_REPLACEMENTS)
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = " ".join(value.split())
    return "".join(c for c in value if 32 <= ord(c) <= 126)


def ascii_name(props, fields):
    """Return an ASCII source name, falling back to transliteration."""
    for key in fields:
        val = props.get(key)
        if not val or not isinstance(val, str):
            continue
        cleaned = " ".join(val.split())
        if not cleaned:
            continue
        if all(32 <= ord(c) <= 126 for c in cleaned):
            return cleaned
        transliterated = transliterate_ascii(cleaned)
        if transliterated:
            return transliterated
    return None


def unwrap_ring(coords, reference_lon=None):
    if not coords:
        return []
    first_lon, first_lat = geographic_coordinate(coords[0])
    result = [[first_lon, first_lat]]
    previous_lon = first_lon
    for coord in coords[1:]:
        lon, lat = geographic_coordinate(coord)
        while lon - previous_lon > 180.0:
            lon -= 360.0
        while lon - previous_lon < -180.0:
            lon += 360.0
        result.append([lon, lat])
        previous_lon = lon
    if reference_lon is not None:
        mean_lon = sum(coord[0] for coord in result) / len(result)
        shift = round((reference_lon - mean_lon) / 360.0) * 360.0
        for coord in result:
            coord[0] += shift
    return result


def ring_area(ring):
    area = 0.0
    for index, point in enumerate(ring):
        previous = ring[index - 1]
        area += previous[0] * point[1] - point[0] * previous[1]
    return area / 2.0


def segment_distance_squared(point, start, end):
    x, y = start
    dx = end[0] - x
    dy = end[1] - y
    if dx != 0.0 or dy != 0.0:
        t = ((point[0] - x) * dx + (point[1] - y) * dy) / (dx * dx + dy * dy)
        if t > 1.0:
            x, y = end
        elif t > 0.0:
            x += dx * t
            y += dy * t
    dx = point[0] - x
    dy = point[1] - y
    return dx * dx + dy * dy


def point_to_polygon_distance(point, rings):
    inside = False
    min_distance_sq = float("inf")
    px, py = point
    for ring in rings:
        for index, current in enumerate(ring):
            previous = ring[index - 1]
            if ((current[1] > py) != (previous[1] > py) and
                    px < (previous[0] - current[0]) * (py - current[1]) /
                    (previous[1] - current[1]) + current[0]):
                inside = not inside
            min_distance_sq = min(
                min_distance_sq,
                segment_distance_squared(point, current, previous),
            )
    distance = math.sqrt(min_distance_sq)
    return distance if inside else -distance


def polygon_centroid(ring):
    area_sum = 0.0
    x_sum = 0.0
    y_sum = 0.0
    for index, point in enumerate(ring):
        previous = ring[index - 1]
        cross = previous[0] * point[1] - point[0] * previous[1]
        x_sum += (previous[0] + point[0]) * cross
        y_sum += (previous[1] + point[1]) * cross
        area_sum += cross
    if abs(area_sum) < 1e-12:
        return ring[0]
    return [x_sum / (3.0 * area_sum), y_sum / (3.0 * area_sum)]


def polygon_label_point(rings, precision=0.02):
    """Mapbox-style polylabel search for an interior polygon anchor."""
    outer = rings[0]
    min_x = min(point[0] for point in outer)
    min_y = min(point[1] for point in outer)
    max_x = max(point[0] for point in outer)
    max_y = max(point[1] for point in outer)
    cell_size = min(max_x - min_x, max_y - min_y)
    if cell_size <= 0.0:
        return outer[0]

    def cell(x, y, half):
        distance = point_to_polygon_distance([x, y], rings)
        return [x, y, half, distance, distance + half * math.sqrt(2.0)]

    queue = []
    serial = 0
    half = cell_size / 2.0
    x = min_x
    while x < max_x:
        y = min_y
        while y < max_y:
            current = cell(x + half, y + half, half)
            heapq.heappush(queue, (-current[4], serial, current))
            serial += 1
            y += cell_size
        x += cell_size

    centroid = polygon_centroid(outer)
    best = cell(centroid[0], centroid[1], 0.0)
    bbox = cell((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, 0.0)
    if bbox[3] > best[3]:
        best = bbox

    while queue:
        _, _, current = heapq.heappop(queue)
        if current[3] > best[3]:
            best = current
        if current[4] - best[3] <= precision:
            continue
        half = current[2] / 2.0
        for dx, dy in ((-half, -half), (half, -half), (-half, half), (half, half)):
            child = cell(current[0] + dx, current[1] + dy, half)
            heapq.heappush(queue, (-child[4], serial, child))
            serial += 1

    return [best[0], best[1]]


def line_length(line):
    length = 0.0
    for index in range(1, len(line)):
        start = geographic_coordinate(line[index - 1])
        end = geographic_coordinate(line[index])
        delta_lon = end[0] - start[0]
        if delta_lon > 180.0:
            delta_lon -= 360.0
        elif delta_lon < -180.0:
            delta_lon += 360.0
        mean_lat = math.radians((start[1] + end[1]) / 2.0)
        length += math.hypot(delta_lon * math.cos(mean_lat), end[1] - start[1])
    return length


def line_midpoint(line):
    total = line_length(line)
    if total <= 0.0:
        return geographic_coordinate(line[0])
    target = total / 2.0
    traversed = 0.0
    for index in range(1, len(line)):
        start = geographic_coordinate(line[index - 1])
        end = geographic_coordinate(line[index])
        delta_lon = end[0] - start[0]
        if delta_lon > 180.0:
            delta_lon -= 360.0
        elif delta_lon < -180.0:
            delta_lon += 360.0
        mean_lat = math.radians((start[1] + end[1]) / 2.0)
        segment_length = math.hypot(
            delta_lon * math.cos(mean_lat), end[1] - start[1]
        )
        if traversed + segment_length >= target:
            fraction = (target - traversed) / segment_length
            return geographic_coordinate([
                start[0] + delta_lon * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            ])
        traversed += segment_length
    return geographic_coordinate(line[-1])


def representative_point(geometry):
    """Pick a reasonable label anchor for Point/Line/Polygon geometry."""
    if not geometry:
        return None
    gtype = geometry.get("type", "")
    coords = geometry.get("coordinates")
    if coords is None:
        return None

    if gtype == "Point":
        return geographic_coordinate(coords)
    if gtype in ("LineString", "MultiLineString"):
        lines = [coords] if gtype == "LineString" else [line for line in coords if line]
        if not lines:
            return None
        return line_midpoint(max(lines, key=line_length))

    polygons = [coords] if gtype == "Polygon" else coords if gtype == "MultiPolygon" else []
    polygons = [polygon for polygon in polygons if polygon and polygon[0]]
    if not polygons:
        return None
    unwrapped = []
    for polygon in polygons:
        outer = unwrap_ring(polygon[0])
        reference_lon = sum(point[0] for point in outer) / len(outer)
        rings = [outer]
        rings.extend(
            unwrap_ring(ring, reference_lon)
            for ring in polygon[1:]
            if ring
        )
        unwrapped.append(rings)
    largest = max(unwrapped, key=lambda rings: abs(ring_area(rings[0])))
    return geographic_coordinate(polygon_label_point(largest))


def geometry_weight(geometry):
    if not geometry:
        return 0.0
    gtype = geometry.get("type", "")
    coords = geometry.get("coordinates") or []
    if gtype == "LineString":
        return line_length(coords)
    if gtype == "MultiLineString":
        return max((line_length(line) for line in coords), default=0.0)
    polygons = [coords] if gtype == "Polygon" else coords if gtype == "MultiPolygon" else []
    return max(
        (abs(ring_area(unwrap_ring(polygon[0]))) for polygon in polygons if polygon),
        default=0.0,
    )


def polygon_label_spine(geometry, point_count=MAX_LABEL_PATH_POINTS):
    """Derive a smooth center path along an elongated polygon footprint."""
    if not geometry or point_count < 2:
        return []
    gtype = geometry.get("type", "")
    coords = geometry.get("coordinates") or []
    polygons = ([coords] if gtype == "Polygon"
                else coords if gtype == "MultiPolygon" else [])

    points = []
    reference_lon = None
    for polygon in polygons:
        if not polygon or not polygon[0]:
            continue
        ring = unwrap_ring(polygon[0], reference_lon)
        if reference_lon is None:
            reference_lon = sum(point[0] for point in ring) / len(ring)
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]
        points.extend(ring)
    if len(points) < point_count * 2:
        return []

    samples = np.asarray(points, dtype=np.float64)
    origin = samples.mean(axis=0)
    centered = samples - origin
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_index = int(np.argmax(eigenvalues))
    major_value = float(eigenvalues[major_index])
    minor_value = float(eigenvalues[1 - major_index])
    if major_value <= 1.0e-8:
        return []
    aspect_ratio = math.sqrt(major_value / max(minor_value, 1.0e-8))
    if aspect_ratio < 2.0:
        return []

    major_axis = eigenvectors[:, major_index]
    minor_axis = np.array([-major_axis[1], major_axis[0]])
    along = centered @ major_axis
    across = centered @ minor_axis
    along_min = float(along.min())
    along_max = float(along.max())
    span = along_max - along_min
    if span <= 1.0e-5:
        return []

    targets = np.linspace(along_min + span * 0.1,
                          along_max - span * 0.1,
                          point_count)
    band_half_width = span * 0.7 / max(point_count - 1, 1)
    across_centers = []
    for target in targets:
        band = across[np.abs(along - target) <= band_half_width]
        if band.size < 2:
            return []
        low, high = np.percentile(band, [10.0, 90.0])
        across_centers.append(float((low + high) * 0.5))

    smoothed = across_centers[:]
    for i in range(1, len(smoothed) - 1):
        smoothed[i] = (across_centers[i - 1] +
                       across_centers[i] * 2.0 +
                       across_centers[i + 1]) * 0.25

    path = []
    for target, cross in zip(targets, smoothed):
        point = origin + major_axis * target + minor_axis * cross
        path.append(geographic_coordinate(point))

    delta_lon = path[-1][0] - path[0][0]
    delta_lat = path[-1][1] - path[0][1]
    if ((abs(delta_lon) >= abs(delta_lat) and delta_lon < 0.0) or
            (abs(delta_lat) > abs(delta_lon) and delta_lat < 0.0)):
        path.reverse()
    return path


def extract_label_record(filename, props, geom):
    """Extract one label record from a feature, or None to skip."""
    if filename == "ne_10m_geography_regions_polys" and props.get("FEATURECLA") == "Continent":
        return None
    if filename == "ne_10m_admin_0_countries":
        if props.get("LABEL_X") is None or props.get("LABEL_Y") is None:
            return None
        pt = geographic_coordinate([props["LABEL_X"], props["LABEL_Y"]])
    elif filename == "ne_10m_admin_1_states_provinces":
        if props.get("longitude") is None or props.get("latitude") is None:
            return None
        pt = geographic_coordinate([props["longitude"], props["latitude"]])
    elif filename == "ne_10m_geography_marine_polys" and \
            props.get("name") in MARINE_LABEL_OVERRIDES:
        pt = MARINE_LABEL_OVERRIDES[props["name"]]
    else:
        pt = representative_point(geom)
    if not pt:
        return None
    lon, lat = pt

    def rec(name, kind, scalerank, min_zoom=0.0, population=0, flags=0,
            dedup_key=None, max_zoom=0.0, label_path=None):
        path = []
        for point in label_path or []:
            path.append(geographic_coordinate(point))
        weight = geometry_weight(geom)
        return {
            "lon": float(np.float32(lon)),
            "lat": float(np.float32(lat)),
            "name": name,
            "kind": kind,
            "scalerank": int(scalerank),
            "min_zoom": float(min_zoom or 0.0),
            "max_zoom": float(max_zoom or 0.0),
            "population": int(population or 0),
            "flags": int(flags),
            "dedup_key": dedup_key,
            "weight": weight,
            "area": weight * math.cos(math.radians(lat)) if kind == KIND_STATE else 0.0,
            "path": path,
        }

    if filename == "ne_10m_populated_places_simple":
        name = ascii_name(props, ["nameascii", "name"])
        if not name:
            return None
        fc = (props.get("featurecla") or "").lower()
        is_capital = "capital" in fc
        flags = 0
        if is_capital:
            flags |= 1
        if props.get("worldcity") == 1:
            flags |= 2
        if props.get("megacity") == 1:
            flags |= 4
        if props.get("adm0cap") == 1:
            flags |= 8
        if fc in ("admin-0 capital", "admin-0 capital alt"):
            flags |= 8
        kind = KIND_CAPITAL if is_capital else KIND_CITY
        return rec(name, kind, props.get("scalerank", 10),
                   props.get("min_zoom", 0), props.get("pop_max", 0), flags)

    if filename == "ne_10m_admin_0_countries":
        # Authoritative country label anchors: use the LABEL_X/LABEL_Y fields
        # Natural Earth computes for label placement (e.g. USA centers on
        # -97.5, 39.5 instead of an Alaska-anchored subunit point).
        # NAME is the compact map label; NAME_EN can contain the full formal name.
        name = ascii_name(props, [
            "NAME", "NAME_EN", "NAME_CIAWF", "ADMIN", "NAME_LONG"
        ])
        if not name:
            return None
        return rec(name, KIND_COUNTRY, props.get("LABELRANK", 0) or 0,
                   props.get("MIN_LABEL", 0), max_zoom=props.get("MAX_LABEL", 0),
                   dedup_key=(props.get("ADM0_A3") or name).lower())

    if filename == "ne_10m_admin_1_states_provinces":
        name = ascii_name(props, ["name", "name_en", "woe_name"])
        if not name:
            return None
        return rec(name, KIND_STATE,
                   props["labelrank"] if props.get("labelrank") is not None else 10,
                   props.get("min_label", 0), max_zoom=props.get("max_label", 0),
                   dedup_key=("state", props.get("adm1_code") or name.lower(),
                              name.lower()))

    if filename == "ne_10m_airports":
        code = ascii_name(props, ["iata_code", "abbrev", "gps_code"])
        name = code or ascii_name(props, ["name"])
        if not name:
            return None
        airport_type = (props.get("type") or "").lower()
        if "major" in airport_type:
            min_zoom = 5.0
            flags = 1
        elif "mid" in airport_type:
            min_zoom = 7.0
            flags = 2
        else:
            min_zoom = 9.0
            flags = 0
        if code:
            flags |= 4
        return rec(name.upper(), KIND_AIRPORT, props.get("scalerank", 10),
                   min_zoom, flags=flags,
                   dedup_key=("airport", props.get("ne_id") or name.lower()))

    if filename == "ne_10m_geography_marine_polys":
        name = ascii_name(props, ["label", "name_en", "name"])
        if not name:
            return None
        scalerank = props.get("scalerank", 9)
        min_label = float(props.get("min_label", 0) or 0)
        if scalerank == 0:
            min_label = max(0.0, min_label - 1.0)
        return rec(name, KIND_WATER, scalerank,
                   min_label,
                   max_zoom=props.get("max_label", 0))

    if filename == "ne_10m_geography_regions_elevation_points":
        name = ascii_name(props, ["name_en", "label", "name"])
        if not name:
            return None
        return rec(name, KIND_PHYSICAL, props.get("scalerank", 9),
                   props.get("min_zoom", 0))

    if filename == "ne_10m_geography_regions_polys":
        name = ascii_name(props, ["LABEL", "NAME_EN", "NAME"])
        if not name:
            return None
        feature_class = props.get("FEATURECLA")
        raw_scalerank = props.get("SCALERANK")
        scalerank = int(9 if raw_scalerank is None else raw_scalerank)
        label_path = []
        if feature_class == "Range/mtn" and scalerank <= 1:
            label_path = polygon_label_spine(geom)
        return rec(name, KIND_REGION, scalerank,
                   props.get("MIN_LABEL", 0),
                   max_zoom=props.get("MAX_LABEL", 0),
                   label_path=label_path)

    if filename == "ne_10m_rivers_lake_centerlines":
        # Prefer `name` over `name_en` (ASCII-filtered): name_en is occasionally
        # a truncated form in this source.
        name = ascii_name(props, ["name", "name_en", "label"])
        if not name:
            return None
        try:
            raw_rank = props.get("scalerank")
            sr = int(round(float(9 if raw_rank is None else raw_rank)))
        except (TypeError, ValueError):
            sr = 9
        return rec(name, KIND_RIVER, sr, props.get("min_label", 0),
                   dedup_key=("river", props.get("dissolve") or
                              props.get("ne_id") or
                              (props.get("rivernum"), name.lower())))

    if filename == "ne_10m_lakes":
        # Prefer `label`/`name` over `name_en`: the source sometimes stores a
        # truncated form in name_en (e.g. "Great Salt" vs "Great Salt Lake").
        name = ascii_name(props, ["label", "name", "name_en"])
        if not name:
            return None
        return rec(name, KIND_LAKE, props.get("scalerank", 9),
                   props.get("min_label", 0))

    return None


# Labels sharing a name with a more authoritative feature at the same place
# (a region polygon named after its country, a city-state's province, the
# municipality of a capital) would draw twice. Drop them here so the runtime
# collision pass never has to choose between two spellings of one name.
LABEL_DUPLICATE_DISTANCE_DEGREES = 2.0
LABEL_DUPLICATE_COUNTRY_DISTANCE_DEGREES = 8.0
LABEL_DUPLICATE_STATE_AREA_DEGREES = 0.25

# Per emitted dataset: (reference dataset, maximum polygon area of the
# candidate in square degrees or None, matching distance in degrees).
LABEL_DUPLICATE_REFERENCES = {
    "ne_10m_admin_1_states_provinces": (
        ("ne_10m_admin_0_countries", None,
         LABEL_DUPLICATE_DISTANCE_DEGREES),
        ("ne_10m_populated_places_simple",
         LABEL_DUPLICATE_STATE_AREA_DEGREES,
         LABEL_DUPLICATE_DISTANCE_DEGREES),
    ),
    "ne_10m_geography_regions_polys": (
        ("ne_10m_admin_0_countries", None,
         LABEL_DUPLICATE_COUNTRY_DISTANCE_DEGREES),
        ("ne_10m_admin_1_states_provinces", None,
         LABEL_DUPLICATE_DISTANCE_DEGREES),
    ),
}

_label_reference_cache = {}


def reference_label_records(filename, source_dir):
    key = (filename, source_dir)
    if key not in _label_reference_cache:
        filepath = os.path.join(source_dir, filename + ".geojson")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        _label_reference_cache[key] = collect_label_records(
            filename, data, source_dir)
    return _label_reference_cache[key]


def drop_duplicate_labels(filename, records, source_dir):
    references = LABEL_DUPLICATE_REFERENCES.get(filename)
    if not references or source_dir is None:
        return records
    by_name = {}
    for reference, max_area, distance in references:
        for rec in reference_label_records(reference, source_dir):
            by_name.setdefault(rec["name"].lower(), []).append(
                (rec, max_area, distance))

    kept = []
    for rec in records:
        duplicate = False
        for other, max_area, distance in by_name.get(rec["name"].lower(), []):
            if max_area is not None and rec["weight"] > max_area:
                continue
            delta_lon = abs(rec["lon"] - other["lon"])
            delta_lon = min(delta_lon, 360.0 - delta_lon)
            if math.hypot(delta_lon, rec["lat"] - other["lat"]) < distance:
                duplicate = True
                break
        if not duplicate:
            kept.append(rec)
    return kept


def collect_label_records(filename, data, source_dir=None):
    features = data.get("features", [])
    if not features:
        features = [{"geometry": data, "properties": {}}]

    records = []
    for feature in features:
        props = feature.get("properties", {}) or {}
        geom = feature.get("geometry", {})
        rec = extract_label_record(filename, props, geom)
        if rec:
            records.append(rec)

    # Keep one real source/geometry anchor per stable feature identity. The
    # point nearest the antimeridian-safe group center avoids synthesizing
    # locations outside multipart states, rivers, and lakes.
    by_key = {}
    groups = {}
    unkeyed = []
    for rec in records:
        key = rec["dedup_key"]
        if key is None:
            unkeyed.append(rec)
            continue
        groups.setdefault(key, []).append(rec)

    for key, group in groups.items():
        if len(group) == 1:
            by_key[key] = group[0]
            continue
        best_rank = min(record["scalerank"] for record in group)
        candidates = [record for record in group
                      if record["scalerank"] == best_rank]
        sin_lon = sum(math.sin(math.radians(record["lon"])) for record in group)
        cos_lon = sum(math.cos(math.radians(record["lon"])) for record in group)
        center_lon = math.degrees(math.atan2(sin_lon, cos_lon))
        center_lat = sum(record["lat"] for record in group) / len(group)

        def center_distance(record):
            delta_lon = abs(record["lon"] - center_lon)
            delta_lon = min(delta_lon, 360.0 - delta_lon)
            return (math.hypot(delta_lon, record["lat"] - center_lat),
                    -record["weight"])

        by_key[key] = min(candidates, key=center_distance)

    return drop_duplicate_labels(filename, unkeyed + list(by_key.values()),
                                 source_dir)


def emit_label_points(fh, filename, filepath):
    print(f"  Extracting label points for {filename}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = collect_label_records(filename, data,
                                    os.path.dirname(os.path.abspath(filepath)))

    out = bytearray()
    out += struct.pack("<I", len(records))
    for rec in records:
        name_b = rec["name"].encode("ascii", "ignore")
        if len(name_b) > MAX_LABEL_NAME:
            raise ValueError(f"Label name exceeds the binary record capacity: {rec['name']!r}")
        out += struct.pack("<ff", rec["lon"], rec["lat"])
        out += struct.pack("<iff", rec["scalerank"], rec["min_zoom"],
                           rec["max_zoom"])
        out += struct.pack("<i", rec["population"])
        out += struct.pack("<f", max(0.0, rec.get("area", 0.0)))
        path = rec.get("path", [])[:MAX_LABEL_PATH_POINTS]
        out += struct.pack("<BBBB", rec["kind"], rec["flags"],
                           len(name_b), len(path))
        out += name_b
        for point in path:
            out += struct.pack("<ff", *point)

    emit_blob(fh, filename + "_labels", bytes(out))

    print(f"    {len(records)} labels")


# Download missing GeoJSON files from Natural Earth.

GEODATA_BASE_URL = "https://cdn.cyberether.org/geodata/"
GEODATA_GH_BASE = (
    "https://raw.githubusercontent.com/nvkelso/"
    "natural-earth-vector/v5.1.2/geojson/"
)
GEODATA_URL_OVERRIDES = {
    "ne_10m_geographic_lines.geojson": GEODATA_GH_BASE + "ne_10m_geographic_lines.geojson",
    "ne_10m_admin_0_countries.geojson": GEODATA_GH_BASE + "ne_10m_admin_0_countries.geojson",
    "ne_10m_admin_1_states_provinces.geojson": GEODATA_GH_BASE + "ne_10m_admin_1_states_provinces.geojson",
    "ne_10m_airports.geojson": GEODATA_GH_BASE + "ne_10m_airports.geojson",
    "ne_10m_geography_marine_polys.geojson": GEODATA_GH_BASE + "ne_10m_geography_marine_polys.geojson",
    "ne_10m_geography_regions_elevation_points.geojson": GEODATA_GH_BASE + "ne_10m_geography_regions_elevation_points.geojson",
    "ne_10m_geography_regions_polys.geojson": GEODATA_GH_BASE + "ne_10m_geography_regions_polys.geojson",
    "ne_10m_admin_0_boundary_lines_disputed_areas.geojson": GEODATA_GH_BASE + "ne_10m_admin_0_boundary_lines_disputed_areas.geojson",
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

if __name__ == "__main__":
    path = sys.argv[1]
    output = sys.argv[2]

    tri_set = set()
    line_set = set()
    label_set = set()
    geo_split_set = set()
    styled_line_set = set()
    inputs = []

    for arg in sys.argv[3:]:
        if arg.startswith("--triangulate="):
            names = arg[len("--triangulate=") :]
            tri_set.update(n.strip() for n in names.split(","))
        elif arg.startswith("--line-segments="):
            names = arg[len("--line-segments=") :]
            line_set.update(n.strip() for n in names.split(","))
        elif arg.startswith("--label-points="):
            names = arg[len("--label-points=") :]
            label_set.update(n.strip() for n in names.split(","))
        elif arg.startswith("--geographic-lines-split="):
            names = arg[len("--geographic-lines-split=") :]
            geo_split_set.update(n.strip() for n in names.split(","))
        elif arg.startswith("--styled-line-segments="):
            names = arg[len("--styled-line-segments=") :]
            styled_line_set.update(n.strip() for n in names.split(","))
        else:
            inputs.append(arg)

    configured = tri_set | line_set | label_set | geo_split_set | styled_line_set
    for filepath in inputs:
        filename = os.path.basename(filepath).split(".")[0]
        if filename not in configured:
            raise ValueError(f"No geodata emitter configured for {filename}")

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
        fh.write("static constexpr uint32_t GeoDataFormatVersion = 2;\n\n")

        for filepath in inputs:
            filename = os.path.basename(filepath).split(".")[0]

            if filename in geo_split_set:
                emit_geographic_lines_split(fh, filename, filepath)
            if filename in tri_set:
                emit_triangulated(fh, filename, filepath)
            if filename in line_set:
                emit_line_segments(fh, filename, filepath)
            if filename in styled_line_set:
                emit_styled_line_segments(fh, filename, filepath)
            if filename in label_set:
                emit_label_points(fh, filename, filepath)

        fh.write("\n}  // namespace Jetstream::Resources\n")
