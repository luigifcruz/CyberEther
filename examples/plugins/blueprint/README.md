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
`-- meson.build
```

`src/plugin.cc` exports the CyberEther plugin ABI symbol. The C++ files register
the block and module using the same registry macros as built-in CyberEther
blocks.

## Build

Install CyberEther so Meson can find `cyberether`, or let Meson fetch the
fallback CyberEther subproject from `subprojects/cyberether.wrap`.

```sh
cd examples/plugins/blueprint
meson setup build
meson compile -C build
```

On Linux, the compiled plugin is written to
`examples/plugins/blueprint/build/cyberether_blueprint_plugin.so`.

The compiled shared module can be loaded with CyberEther's plugin loader.

When building CyberEther from the repository root with examples enabled, the
same plugin is written to `build-release/examples/plugins/cyberether_blueprint_plugin.so`.
