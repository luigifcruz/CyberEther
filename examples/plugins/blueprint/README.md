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
|   `-- gain.yml
|-- src/
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
|   `-- bundler.py
|-- meson.build
`-- meson_options.txt
```

The plugin ID is `blueprint`, so every block type and implementation it
registers starts with `blueprint_`. The build compiles one library for the
CyberEther build it is configured against, then packages that library,
`manifest.yml`, and the bundled examples into a `.cep` plugin bundle using
`tools/bundler.py`. Each bundle holds one CyberEther build, so build once per
platform to ship several.

## Build

Install CyberEther so Meson can find `cyberether`, or let Meson fetch the
fallback CyberEther subproject from `subprojects/cyberether.wrap`.

```sh
cd examples/plugins/blueprint
meson setup build
meson compile -C build
meson test -C build --suite plugin --print-errorlogs
```

The compiled plugin bundle is written to
`build/blueprint-<version>-<build-id>.cep`. The build also creates separate
module and block test executables. Disable them with
`meson setup build -Dtests=false` when Catch2 is not needed.

The plugin is built for the exact CyberEther version it compiles against, and
for that CyberEther build, such as `macos-arm64` or `linux-x86_64-cuda`. The
CyberEther dependency exports its build ID and enabled backends, and the build
compiles each provider source only when that backend is available. When adding
another backend, add its provider source beside `module_impl_native_cpu.cc`
behind a check on `cyberether_backends`.

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

Install the `.cep` bundle from **Settings** > **Registry**, or load it for one
run with `cyberether --plugin`.

For more details, see the [CyberEther plugin documentation](https://cyberether.org/docs/plugins).
