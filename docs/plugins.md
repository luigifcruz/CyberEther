---
title: Creating Plugins
description: How to create external CyberEther plugins.
order: 80
category: Development
---

CyberEther plugins are shared libraries that register blocks and modules when
loaded. They are useful for keeping custom processing blocks outside the main
CyberEther source tree while still using CyberEther's block, module, scheduler,
runtime, and memory APIs.

## Starting From The Blueprint

The repository includes a plugin blueprint at:

```text
examples/plugins/blueprint
```

Copy that directory when starting a new plugin, then rename the project,
include folder, source folder, block type, module type, and plugin name.

## Folder Layout

The blueprint uses the same block and module layout as built-in CyberEther
blocks:

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
|-- subprojects/
|   `-- cyberether.wrap
`-- meson.build
```

The public headers in `include/` define the block and module configuration.
The source files in `src/` implement the block, implement the module, and
register the native CPU provider.

## Plugin ABI

Every plugin must export CyberEther's plugin ABI symbol. In the blueprint this
lives in `src/plugin.cc`:

```cpp
#include <jetstream/plugin.hh>

JST_REGISTER_PLUGIN();
```

The ABI record only identifies the shared library as a compatible CyberEther
plugin ABI.

| Field | Purpose |
|-------|---------|
| Magic | Identifies the exported record as CyberEther's plugin ABI. |
| Size | Size of the ABI record exported by the plugin. |
| ABI version | Plugin ABI version expected by CyberEther. |

## Blocks And Modules

A block is the user-facing graph node. It defines the block type, domain,
description, configuration fields, inputs, and outputs.

A module does the runtime work. The blueprint includes a `BlueprintGain` module
with a native CPU implementation that accepts `F32` and `CF32` tensors.

The key registration points are:

```cpp
JST_REGISTER_BLOCK(BlueprintGainImpl);
```

and:

```cpp
JST_REGISTER_MODULE(BlueprintGainImplNativeCpu,
                    DeviceType::CPU,
                    RuntimeType::NATIVE,
                    "generic");
```

When CyberEther loads the plugin, those static registrations are drained into
the CyberEther registry.

## Building Standalone

Build the blueprint as a standalone plugin from its own directory:

```sh
cd examples/plugins/blueprint
meson setup build
meson compile -C build
```

On Linux, the output is:

```text
examples/plugins/blueprint/build/cyberether_blueprint_plugin.so
```

The blueprint includes `subprojects/cyberether.wrap`, so Meson can fetch
CyberEther as a fallback when it cannot find an installed CyberEther dependency.

## Building From The CyberEther Tree

When building CyberEther itself with examples enabled, the blueprint is also
available as a root build target:

```sh
meson compile -C build-release cyberether_blueprint_plugin
```

On Linux, the output is:

```text
build-release/examples/plugins/cyberether_blueprint_plugin.so
```

The `.so.p` directories in the build tree are Meson's private object directories.
They are not the final plugin shared library.

## Loading A Plugin

CyberEther loads plugins through its plugin loader. At load time, CyberEther:

1. Opens the shared library.
2. Looks up the exported plugin ABI symbol.
3. Validates ABI magic, size, and ABI version.
4. Drains the plugin's static block and module registrations into the registry.

After the plugin is loaded, its registered blocks can be built like other
CyberEther blocks.
