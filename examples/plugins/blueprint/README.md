# CyberEther Plugin Blueprint

This is a standalone CyberEther plugin blueprint. Copy this directory, rename
`blueprint`, and replace the `gain` block with your own block and module.

## Layout

```text
.
|-- include/
|   `-- blueprint/
|       `-- gain/
|           |-- block.hh
|           `-- module.hh
|-- examples/
|   `-- blueprint_gain.yml
|-- src/
|   |-- plugin.cc
|   |-- meson.build
|   `-- blueprint/
|       `-- gain/
|           |-- block_impl.cc
|           |-- block_tests.cc
|           |-- meson.build
|           |-- module_benchmarks.cc
|           |-- module_impl.cc
|           |-- module_impl.hh
|           |-- module_impl_native_cpu.cc
|           `-- module_tests.cc
|-- subprojects/
|   |-- catch2.wrap
|   `-- cyberether.wrap
|-- tests/
|   |-- main.cc
|   `-- meson.build
|-- tools/
|   |-- bundler.py
|   `-- merger.py
|-- meson.build
`-- meson_options.txt
```

The `src/plugin.cc` exports the CyberEther plugin ABI symbol for the target shared
library. The build packages that library, `manifest.yml`, and bundled examples
into a `.cep` plugin bundle using `tools/bundler.py`. Single-target bundles built
on separate machines can be combined into one multi-target bundle with
`tools/merger.py`.

## Build

Install CyberEther so Meson can find `cyberether`, or let Meson fetch the
fallback CyberEther subproject from `subprojects/cyberether.wrap`.

```sh
cd examples/plugins/blueprint
meson setup build
meson compile -C build
meson test -C build --suite plugin --print-errorlogs
```

The compiled plugin bundle is written to `build/cyberether_blueprint_plugin.cep`.
The build also creates separate module and block test executables. Disable them
with `meson setup build -Dtests=false` when Catch2 is not needed.

For the browser, build with Emscripten and the same CyberEther cross file
used by the host so their dynamic-linking settings match:

```sh
meson setup build-wasm \
  --cross-file ../../../meson/crosscompile/emscripten.ini \
  -Dbuildtype=release \
  -Dtests=false
meson compile -C build-wasm cyberether_blueprint_plugin_cep
```

This produces a `.cep` bundle containing a `browser-wasm32` WebAssembly side
module. When using a copied blueprint, adjust the cross-file path to the
CyberEther source tree used to build the browser host.

Tests follow CyberEther's module convention and live beside each component. Add
`module_tests.cc` or `block_tests.cc` to that component's `plugin_test_lst`; the
shared `tests/meson.build` creates and registers a separate executable for every
listed source.

Module benchmarks use `JST_BENCHMARKS` and compile into the plugin alongside the
implementation. The gain example registers F32 and CF32 cases at several input
sizes for CyberEther's benchmark runner.

Provider `validate()` hooks own backend capability checks before `create()` has
side effects. Missing, malformed, and empty inputs remain framework-owned so
provider validation should defer them to the common lifecycle checks.

The `.cep` bundle can be loaded with CyberEther's plugin loader.

For more details, see the [CyberEther plugin documentation](https://cyberether.org/docs/plugins).
