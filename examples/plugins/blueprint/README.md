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
|           |-- meson.build
|           |-- module_impl.cc
|           |-- module_impl.hh
|           `-- module_impl_native_cpu.cc
|-- tools/
|   `-- bundler.py
`-- meson.build
```

The `src/plugin.cc` exports the CyberEther plugin ABI symbol for the target shared
library. The build packages that library, `manifest.yml`, and bundled examples
into a `.cep` plugin bundle using `tools/bundler.py`.

## Build

Install CyberEther so Meson can find `cyberether`, or let Meson fetch the
fallback CyberEther subproject from `subprojects/cyberether.wrap`.

```sh
cd examples/plugins/blueprint
meson setup build
meson compile -C build
```

The compiled plugin bundle is written to `build/cyberether_blueprint_plugin.cep`.

The `.cep` bundle can be loaded with CyberEther's plugin loader.

For more details, see the [CyberEther plugin documentation](https://cyberether.org/docs/plugins).
