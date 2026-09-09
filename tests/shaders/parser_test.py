#!/usr/bin/env python3

import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "shader_parser", ROOT / "resources" / "shaders" / "parser.py"
)
PARSER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARSER)


class ShaderParserTests(unittest.TestCase):
    def test_only_declared_shaders_and_backends_are_packaged(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "resources" / "shaders"
            output.mkdir(parents=True)
            inputs = []
            for stage in ("vert", "frag"):
                current = output / f"map_geoline.{stage}.spv"
                current.write_bytes(b"current")
                inputs.append(str(current))
                # Old names AND old backend variants must not survive a rebuild.
                (output / f"map_ocean.{stage}.spv").write_bytes(b"stale")
                (output / f"map_geoline.{stage}.msl").write_bytes(b"stale")
            PARSER.bin_to_header(directory, "map", inputs)
            header = (output / "map_shaders.hh").read_text()
            self.assertIn('"geoline"', header)
            self.assertIn("DeviceType::Vulkan", header)
            self.assertNotIn("ocean", header)
            self.assertNotIn("DeviceType::Metal", header)
            PARSER.bin_to_header(directory, "map", list(reversed(inputs)))
            self.assertEqual((output / "map_shaders.hh").read_text(), header)

    def test_missing_fragment_and_duplicate_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete shader pair"):
            PARSER.collect_inputs(["map_fill.vert.spv"], "map")
        with self.assertRaisesRegex(ValueError, "Duplicate shader input"):
            PARSER.collect_inputs(["map_fill.vert.spv"] * 2, "map")

    def test_global_kernels_keep_names_and_per_entry_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "resources" / "shaders"
            output.mkdir(parents=True)
            inputs = []
            for name, target in (("thick_lines", "spv"), ("shapes", "msl")):
                path = output / f"global_{name}.comp.{target}"
                path.write_bytes(b"kernel")
                inputs.append(str(path))
            PARSER.bin_to_header(directory, "global", inputs)
            header = (output / "global_shaders.hh").read_text()
            self.assertIn("GlobalShadersPackage", header)
            self.assertIn("GlobalKernelsPackage", header)
            self.assertIn("thick_lines_spv_kernel", header)
            self.assertIn("shapes_msl_kernel", header)
            self.assertNotIn("thick_lines_msl_kernel", header)
            self.assertNotIn("shapes_spv_kernel", header)


if __name__ == "__main__":
    unittest.main()
