#!/usr/bin/env python3

import importlib.util
import gzip
import io
import json
import pathlib
import re
import runpy
import struct
import tempfile
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "geodata_parser", ROOT / "resources" / "geodata" / "parser.py"
)
PARSER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARSER)


class GeoDataParserTests(unittest.TestCase):
    @staticmethod
    def state_feature(name, code, lon, rank=2):
        return {
            "properties": {
                "name": name, "adm1_code": code, "labelrank": rank,
                "longitude": lon, "latitude": -16,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[lon - 0.1, -16.1], [lon + 0.1, -16.1],
                                 [lon + 0.1, -15.9], [lon - 0.1, -15.9],
                                 [lon - 0.1, -16.1]]],
            },
        }

    def test_transliterates_global_names(self):
        self.assertEqual(
            PARSER.ascii_name({"name": "São Paulo"}, ["name"]),
            "Sao Paulo",
        )
        self.assertEqual(
            PARSER.ascii_name({"name": "Łódź"}, ["name"]),
            "Lodz",
        )
        self.assertEqual(
            PARSER.ascii_name({"name": "Bakı"}, ["name"]),
            "Baki",
        )
        self.assertEqual(
            PARSER.ascii_name(
                {"name": "Pääjärvi", "name_en": "Pyaozero"},
                ["name", "name_en"],
            ),
            "Paajarvi",
        )

    def test_polygon_anchor_is_inside_concave_polygon(self):
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [0, 0], [6, 0], [6, 1], [1, 1],
                [1, 5], [6, 5], [6, 6], [0, 6], [0, 0],
            ]],
        }
        point = PARSER.representative_point(geometry)
        rings = [PARSER.unwrap_ring(geometry["coordinates"][0])]
        self.assertGreater(PARSER.point_to_polygon_distance(point, rings), 0.0)

    def test_state_dedup_is_antimeridian_safe_and_identity_based(self):
        records = PARSER.collect_label_records(
            "ne_10m_admin_1_states_provinces",
            {"features": [
                self.state_feature("Northern", "FJI-1", 179.5),
                self.state_feature("Northern", "FJI-1", -179.5),
                self.state_feature("Northern", "FJI-2", 50.0),
            ]},
        )
        self.assertEqual(len(records), 2)
        fiji = next(record for record in records
                    if abs(record["lon"]) > 170.0)
        self.assertGreater(abs(fiji["lon"]), 170.0)

    def test_namesake_rivers_keep_separate_feature_identities(self):
        def river(rivernum, lon):
            return {
                "properties": {
                    "name": "Mackenzie",
                    "rivernum": rivernum,
                    "scalerank": 2,
                    "min_label": 3.0,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, 10], [lon + 1, 10]],
                },
            }

        records = PARSER.collect_label_records(
            "ne_10m_rivers_lake_centerlines",
            {"features": [river(34, -130), river(785, 148)]},
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(sorted(record["min_zoom"] for record in records),
                         [3.0, 3.0])

    def test_reused_source_ids_do_not_merge_different_names(self):
        def river(name):
            return {
                "properties": {
                    "name": name,
                    "rivernum": 303,
                    "scalerank": 2,
                    "min_label": 3.0,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0, 0], [1, 0]],
                },
            }

        records = PARSER.collect_label_records(
            "ne_10m_rivers_lake_centerlines",
            {"features": [river("Garonne"), river("Drau")]},
        )
        self.assertEqual({record["name"] for record in records},
                         {"Garonne", "Drau"})

    def test_state_dedup_prefers_lowest_label_rank(self):
        records = PARSER.collect_label_records(
            "ne_10m_admin_1_states_provinces",
            {"features": [
                self.state_feature("Example State", "EXP-1", -120, 4),
                self.state_feature("Example State", "EXP-1", -118, 1),
                self.state_feature("Example Capital", "EXP-1", -118, 1),
            ]},
        )
        self.assertEqual(len(records), 2)
        state_record = next(record for record in records
                            if record["name"] == "Example State")
        self.assertEqual(state_record["lon"], -118)
        self.assertEqual(state_record["scalerank"], 1)
        self.assertGreater(state_record["area"], 0)

    def test_uses_label_visibility_and_iata_metadata(self):
        country = PARSER.extract_label_record(
            "ne_10m_admin_0_countries",
            {
                "NAME": "Example",
                "ADM0_A3": "EXP",
                "LABEL_X": 10,
                "LABEL_Y": 20,
                "LABELRANK": 2,
                "MIN_LABEL": 1.7,
                "MAX_LABEL": 6.0,
            },
            {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 1], [0, 0]]]},
        )
        airport = PARSER.extract_label_record(
            "ne_10m_airports",
            {
                "iata_code": "TKG",
                "abbrev": "WIIT",
                "type": "major",
                "scalerank": 4,
                "ne_id": 1,
            },
            {"type": "Point", "coordinates": [10, 20]},
        )
        self.assertEqual(country["min_zoom"], 1.7)
        self.assertEqual(country["max_zoom"], 6.0)
        self.assertEqual(airport["name"], "TKG")
        self.assertEqual(airport["min_zoom"], 5.0)
        self.assertEqual(airport["flags"], 5)

    def test_physical_regions_keep_label_range(self):
        region = PARSER.extract_label_record(
            "ne_10m_geography_regions_polys",
            {
                "NAME": "ALPS",
                "NAME_EN": "Alps",
                "LABEL": "ALPS",
                "FEATURECLA": "Range/mtn",
                "SCALERANK": 1,
                "MIN_LABEL": 2.0,
                "MAX_LABEL": 8.0,
                "NE_ID": 1,
            },
            {
                "type": "Polygon",
                "coordinates": [[[5, 45], [15, 45], [10, 48], [5, 45]]],
            },
        )
        self.assertEqual(region["kind"], PARSER.KIND_REGION)
        self.assertEqual(region["name"], "ALPS")
        self.assertEqual(region["min_zoom"], 2.0)
        self.assertEqual(region["max_zoom"], 8.0)

    def test_elongated_major_range_gets_a_label_spine(self):
        ring = ([[0, latitude] for latitude in range(21)] +
                [[2, latitude] for latitude in reversed(range(21))] +
                [[0, 0]])
        geometry = {
            "type": "Polygon",
            "coordinates": [ring],
        }
        region = PARSER.extract_label_record(
            "ne_10m_geography_regions_polys",
            {
                "NAME": "TEST RANGE",
                "FEATURECLA": "Range/mtn",
                "SCALERANK": 1,
                "MIN_LABEL": 2,
                "MAX_LABEL": 6,
            },
            geometry,
        )
        self.assertEqual(len(region["path"]), PARSER.MAX_LABEL_PATH_POINTS)
        self.assertLess(region["path"][0][1], region["path"][-1][1])

    def test_continents_are_excluded_from_label_resources(self):
        continent = PARSER.extract_label_record(
            "ne_10m_geography_regions_polys",
            {
                "NAME": "TEST CONTINENT",
                "FEATURECLA": "Continent",
                "SCALERANK": 0,
                "MIN_LABEL": 0,
                "MAX_LABEL": 4,
            },
            {
                "type": "Polygon",
                "coordinates": [[[0, 0], [10, 0], [0, 10], [0, 0]]],
            },
        )
        self.assertIsNone(continent)

    def test_unclassified_inputs_fail_before_download_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "unsupported.geojson"
            output = pathlib.Path(directory) / "geodata.hh"
            parser = ROOT / "resources" / "geodata" / "parser.py"
            with patch("sys.argv", [str(parser), directory, str(output), str(source)]):
                with self.assertRaisesRegex(ValueError, "No geodata emitter configured"):
                    runpy.run_path(str(parser), run_name="__main__")
            self.assertFalse(source.exists())
            self.assertFalse(output.exists())

    def test_river_geometry_is_split_by_rank(self):
        self.assertEqual(PARSER.river_line_category({"scalerank": 0}), 0)
        self.assertEqual(PARSER.river_line_category({"scalerank": 3}), 3)
        self.assertEqual(PARSER.river_line_category({"scalerank": 4}), 4)
        self.assertEqual(PARSER.river_line_category({"scalerank": 10}), 10)
        self.assertEqual(PARSER.river_line_category({}), 10)
        self.assertEqual(PARSER.river_line_category({"scalerank": "bad"}), 10)

    def test_southern_ocean_basin_labels_clear_the_tropic(self):
        self.assertEqual(
            PARSER.MARINE_LABEL_OVERRIDES["South Atlantic Ocean"],
            (-15.0, -35.0),
        )
        self.assertEqual(
            PARSER.MARINE_LABEL_OVERRIDES["South Pacific Ocean"],
            (-130.0, -35.0),
        )

    def test_triangulation_is_sphere_safe_across_antimeridian(self):
        # Fills render as straight 3D chords on the unit globe; triangle
        # edges spanning many degrees dip inside the sphere and open holes
        # near the limb. Rings crossing the antimeridian must additionally
        # keep their triangulation topology contiguous.
        import math

        def sphere(lon, lat):
            r = math.radians(lat)
            lr = math.radians(lon)
            cl = math.cos(r)
            return (cl * math.sin(lr), math.sin(r), cl * math.cos(lr))

        def arc_degrees(a, b):
            cross = (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )
            cross_norm = math.sqrt(sum(c * c for c in cross))
            dot = sum(x * y for x, y in zip(a, b))
            return math.degrees(math.atan2(cross_norm, dot))

        ring = [
            [150.0, -20.0], [-160.0, -20.0], [-160.0, 30.0],
            [150.0, 30.0], [150.0, -20.0],
        ]
        geographic, indices = PARSER.triangulate_polygon([ring])
        self.assertTrue(geographic)
        self.assertEqual(len(indices) % 3, 0)
        for lon, lat in zip(geographic[0::2], geographic[1::2]):
            self.assertGreaterEqual(lon, -180.0)
            self.assertLessEqual(lon, 180.0)
            self.assertGreaterEqual(lat, -90.0)
            self.assertLessEqual(lat, 90.0)
        points = [
            sphere(geographic[i], geographic[i + 1])
            for i in range(0, len(geographic), 2)
        ]
        for t in range(0, len(indices), 3):
            a, b, c = indices[t:t + 3]
            longest = max(
                arc_degrees(points[a], points[b]),
                arc_degrees(points[b], points[c]),
                arc_degrees(points[c], points[a]),
            )
            self.assertLessEqual(
                longest,
                PARSER.MAX_FILL_EDGE_DEGREES + 1e-3,
            )

    def emit_resource(self, emitter, name, features):
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / (name + ".geojson")
            source.write_text(json.dumps({"features": features}))
            output = io.StringIO()
            emitter(output, name, str(source))
        header = output.getvalue()
        compressed = bytes(int(value, 16) for value in re.findall(r"0x([0-9a-f]{2})", header))
        raw = gzip.decompress(compressed)
        self.assertIn(f"_gz_len = {len(compressed)};", header)
        self.assertIn(f"_raw_len = {len(raw)};", header)
        return raw

    def test_line_binary_has_only_geographic_endpoints(self):
        raw = self.emit_resource(PARSER.emit_line_segments, "line", [{
            "geometry": {"type": "LineString", "coordinates": [[179, 10], [-179, 12]]},
        }])
        count, = struct.unpack_from("<I", raw)
        self.assertEqual(count, 8)
        self.assertEqual(len(raw), 4 + count * 4)
        self.assertEqual(struct.unpack_from("<8f", raw, 4),
                         (179, 10, 180, 11, -180, 11, -179, 12))

    def test_fill_binary_retains_indices_without_mercator_stream(self):
        rings = [[[0, 0], [1, 0], [0, 1], [0, 0]]]
        geographic, indices = PARSER.triangulate_polygon(rings)
        raw = self.emit_resource(PARSER.emit_triangulated, "fill", [{
            "geometry": {"type": "Polygon", "coordinates": rings},
        }])
        vertices, count = struct.unpack_from("<II", raw)
        self.assertEqual(vertices * 2, len(geographic))
        self.assertEqual(count, len(indices))
        self.assertEqual(len(raw), 8 + vertices * 8 + count * 4)
        self.assertEqual(struct.unpack_from(f"<{vertices * 2}f", raw, 8), tuple(geographic))
        self.assertEqual(struct.unpack_from(f"<{count}I", raw, 8 + vertices * 8), tuple(indices))

    def test_styled_binary_retains_zoom_and_rank(self):
        raw = self.emit_resource(PARSER.emit_styled_line_segments,
                                 "ne_10m_rivers_lake_centerlines", [{
            "properties": {"scalerank": 4, "min_zoom": 3},
            "geometry": {"type": "LineString", "coordinates": [[10, 20], [11, 21]]},
        }])
        self.assertEqual(len(raw), 4 + 16 + 4 + 1)
        self.assertEqual(struct.unpack("<I5fB", raw), (1, 10, 20, 11, 21, 3, 4))

    def test_label_binary_preserves_long_country_names(self):
        names = [
            "People's Republic of China",
            "Democratic Republic of the Congo",
            "United Kingdom of Great Britain and Northern Ireland",
        ]
        raw = self.emit_resource(PARSER.emit_label_points, "ne_10m_admin_0_countries", [{
            "properties": {"NAME_EN": name, "LABEL_X": 10, "LABEL_Y": 20},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 1], [0, 0]]]},
        } for name in names])
        self.assertEqual(struct.unpack_from("<I", raw)[0], len(names))
        offset = 4
        decoded = []
        for _ in names:
            chars, points = raw[offset + 30:offset + 32]
            offset += 32
            decoded.append(raw[offset:offset + chars].decode("ascii"))
            offset += chars + points * 8
        self.assertEqual(decoded, names)
        self.assertEqual(offset, len(raw))

    def test_label_binary_retains_metadata_and_geographic_paths(self):
        ring = ([[0, latitude] for latitude in range(21)] +
                [[2, latitude] for latitude in reversed(range(21))] + [[0, 0]])
        raw = self.emit_resource(PARSER.emit_label_points,
                                 "ne_10m_geography_regions_elevation_points", [{
            "properties": {"name_en": "PEAK", "scalerank": 1, "min_zoom": 2},
            "geometry": {"type": "Point", "coordinates": [10, 20]},
        }])
        self.assertEqual(struct.unpack_from("<I", raw)[0], 1)
        lon, lat, rank, minimum, maximum, population, area, kind, flags, chars, points = \
            struct.unpack_from("<ffiffifBBBB", raw, 4)
        self.assertEqual((lon, lat, rank, minimum, maximum, population, area),
                         (10, 20, 1, 2, 0, 0, 0))
        self.assertEqual((kind, flags, chars, points), (PARSER.KIND_PHYSICAL, 0, 4, 0))
        self.assertEqual(raw[36:], b"PEAK")

        # Regions have cross-dataset dedup references; inspect their path records
        # directly so this fixture does not need the full Natural Earth sources.
        record = PARSER.extract_label_record("ne_10m_geography_regions_polys", {
            "NAME": "RANGE", "FEATURECLA": "Range/mtn", "SCALERANK": 1,
        }, {"type": "Polygon", "coordinates": [ring]})
        self.assertEqual(len(record["path"]), PARSER.MAX_LABEL_PATH_POINTS)
        self.assertTrue(all(len(point) == 2 for point in record["path"]))
        self.assertNotIn("mercX", record)
        self.assertNotIn("mercY", record)
        with patch.object(PARSER, "drop_duplicate_labels", side_effect=lambda _, records, __: records):
            raw = self.emit_resource(PARSER.emit_label_points, "ne_10m_geography_regions_polys", [{
                "properties": {"NAME": "RANGE", "FEATURECLA": "Range/mtn", "SCALERANK": 1},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }])
        chars, points = raw[34:36]
        self.assertEqual((chars, points), (5, PARSER.MAX_LABEL_PATH_POINTS))
        self.assertEqual(len(raw), 4 + 32 + chars + points * 8)
        self.assertEqual(struct.unpack_from(f"<{points * 2}f", raw, 36 + chars),
                         tuple(value for point in record["path"] for value in point))


if __name__ == "__main__":
    unittest.main()
