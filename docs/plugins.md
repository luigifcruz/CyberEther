---
title: Creating Plugins
description: How to create external CyberEther plugins.
order: 82
category: Development
---

> [!WARNING]
>
> Plugin bundles are experimental. The format, ABI, tooling, and loader behavior can change at any time.

CyberEther plugins are `.cep` bundles that contain one or more shared libraries,
a manifest, and optional example flowgraphs. They are useful for keeping custom
processing blocks outside the main CyberEther source tree while still using
CyberEther's block, module, scheduler, runtime, and memory APIs.

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
|-- subprojects/
|   `-- cyberether.wrap
|-- tools/
|   `-- bundler.py
`-- meson.build
```

The public headers in `include/` define the block and module configuration.
The source files in `src/` implement the block, implement the module, and
register the native CPU provider. Files in `examples/` are bundled as plugin
examples. The `tools/bundler.py` script creates the `.cep` bundle for the copied
blueprint.

## CEP Bundles

A `.cep` file is a `tar.gz` archive with a `.cep` extension. It must include a
`manifest.yml` at the archive root:

```yaml
metadata:
  name: cyberether-blueprint-plugin
  version: 0.1.0
  minimumJetstreamVersion: 1.6.0

targets:
  - path: targets/macos-arm64-cpu/cyberether_blueprint_plugin.dylib
    system: macos
    device: cpu
    arch: arm64

examples:
  - path: examples/blueprint_gain.yml
```

| Field | Purpose |
|-------|---------|
| `metadata.name` | Plugin bundle name. |
| `metadata.version` | Plugin bundle version. |
| `metadata.minimumJetstreamVersion` | Minimum CyberEther/Jetstream version required to load the bundle. |
| `targets[].path` | Shared library path inside the bundle. |
| `targets[].system` | Target system, such as `macos`, `linux`, or `windows`. |
| `targets[].device` | Device backend, such as `cpu`, `cuda`, `metal`, `vulkan`, or `webgpu`. |
| `targets[].arch` | Target architecture, such as `arm64` or `x86_64`. |
| `examples[].path` | Example flowgraph path inside the bundle. |

CyberEther loads every target that matches the current system, architecture,
and compiled device backends. Development builds usually package one target;
release automation can package multiple systems, architectures, and devices in
the same `.cep`.

## Plugin ABI

Every target shared library must export CyberEther's plugin ABI symbol. In the
blueprint this lives in `src/plugin.cc`:

```cpp
#include <jetstream/plugin.hh>

JST_REGISTER_PLUGIN();
```

The ABI record only identifies the target library as a compatible CyberEther
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

When CyberEther loads a compatible target from the bundle, those static
registrations are drained into the CyberEther registry.

## Dependencies

A plugin bundle is copied to machines you do not control, so its shared
libraries should load without extra setup. The recommendation is to depend only
on libraries that are commonly present on the target system, such as the C and
C++ standard libraries, and to link everything else statically into the plugin
library.

If static linking is not possible for some dependency, the plugin documentation
must clearly list every external dependency the user needs to install, including
the expected version range and the package name on each supported system. A
plugin that fails to load because of a missing shared library is hard for users
to diagnose, so treat undocumented runtime dependencies as a packaging bug.

## Bundling

Use the blueprint's `tools/bundler.py` to create `.cep` files. From your copied
blueprint directory:

```sh
./tools/bundler.py \
  --output build/cyberether_blueprint_plugin.cep \
  --name cyberether-blueprint-plugin \
  --version 0.1.0 \
  --minimum-jetstream-version 1.6.0 \
  --target path=build/cyberether_blueprint_plugin.dylib,system=macos,device=cpu,arch=arm64 \
  --example examples/blueprint_gain.yml
```

Repeat `--target` for production bundles that include multiple compatible
libraries. Repeat `--example` to include more example flowgraphs.

## Building Standalone

Build the blueprint as a standalone plugin from its own directory:

```sh
cd examples/plugins/blueprint
meson setup build
meson compile -C build
```

On Linux, the output is:

```text
build/cyberether_blueprint_plugin.cep
```

The blueprint includes `subprojects/cyberether.wrap`, so Meson can fetch
CyberEther as a fallback when it cannot find an installed CyberEther dependency.

## Building From The CyberEther Tree

When building CyberEther itself with examples enabled, the blueprint is also
available as a root build target:

```sh
meson compile -C build-release cyberether_blueprint_plugin_cep
```

On Linux, the output is:

```text
build-release/examples/plugins/cyberether_blueprint_plugin.cep
```

The shared library in the build tree is an intermediate target. The `.cep` file
is the user-facing plugin artifact.

## Loading A Plugin

CyberEther loads plugins through its plugin loader. At load time, CyberEther:

1. Copies the `.cep` bundle into the plugin cache.
2. Extracts the bundled `tar.gz` into a cache folder.
3. Reads and validates `manifest.yml`.
4. Selects targets matching the current system, architecture, and device support.
5. Opens every compatible shared library.
6. Looks up the exported plugin ABI symbol for each target.
7. Validates ABI magic, size, and ABI version.
8. Drains static block and module registrations into the registry.
9. Registers bundled examples from `examples[].path`.

After the plugin is loaded, its registered blocks can be built like other
CyberEther blocks. The user-facing side of this flow, including registration
through the preferences window, is covered in
[Installing Plugins](/docs/installing-plugins).
