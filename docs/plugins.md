---
title: Creating Plugins
description: How to create external CyberEther plugins.
order: 360
category: Development
---

Plugins let you develop, build, and distribute custom CyberEther blocks outside the main source tree while using the exact same block, module, scheduler, runtime, and tensor APIs as built-in blocks. Each plugin ships as a `.cep` bundle containing compiled shared libraries, a manifest, and optional example flowgraphs.

> [!WARNING]
> Plugin bundles are experimental. The bundle format, ABI, tooling, and loader behavior may change between CyberEther releases.

## Starting From the Blueprint

The fastest way to start a new plugin is to copy the official blueprint included in the repository:

```text
examples/plugins/blueprint
```

After copying the directory, rename the Meson project, include folder, source folder, block type, module type, and plugin name to match your own plugin.

### Blueprint Layout

The blueprint mirrors the standard built-in block layout:

```text
.
├── include/
│   └── blueprint/
│       └── gain/
│           ├── block.hh                 Public Block Config
│           └── module.hh                Public Module Config
├── examples/
│   └── blueprint_gain.yml               Bundled example flowgraph
├── src/
│   ├── plugin.cc                        Exports the Plugin ABI symbol
│   ├── meson.build
│   └── blueprint/
│       └── gain/
│           ├── block_impl.cc            Block implementation & registration
│           ├── meson.build
│           ├── module_impl.cc           Shared module logic
│           ├── module_impl.hh
│           └── module_impl_native_cpu.cc  Native CPU kernel & registration
├── subprojects/
│   └── cyberether.wrap                  Fallback Meson wrap for CyberEther
├── tools/
│   ├── bundler.py                       Packages single/multi-target .cep files
│   └── merger.py                        Combines platform .cep files for release
├── meson.build
└── meson_options.txt
```

## Exporting the Plugin ABI

Every shared library inside a `.cep` bundle must export CyberEther's plugin ABI symbol. In the blueprint, `src/plugin.cc` handles this in two lines:

```cpp
#include <jetstream/plugin.hh>

JST_REGISTER_PLUGIN();
```

The exported record lets the loader verify compatibility before touching any static registrations:

| Field | Purpose |
|---|---|
| **Magic** | Identifies the library as a valid CyberEther plugin. |
| **Size** | Byte size of the exported ABI record. |
| **ABI version** | Plugin ABI version expected by CyberEther. |

## Registering Blocks and Modules

Plugins use the exact same registration macros as built-in blocks (see [Blocks And Modules](/docs/blocks-and-modules)). The blueprint implements a `BlueprintGain` block backed by a native CPU module supporting `F32` and `CF32` tensors:

```cpp
JST_REGISTER_BLOCK(BlueprintGainImpl, {"blueprint_gain"});

JST_REGISTER_MODULE(BlueprintGainImplNativeCpu,
                    DeviceType::CPU,
                    RuntimeType::NATIVE,
                    "generic");
```

When CyberEther opens a compatible shared library from your bundle, those static registrations drain straight into the global registry.

## Anatomy of a `.cep` Bundle

A `.cep` file is a standard `tar.gz` archive with a `.cep` extension containing a root `manifest.yml`:

```yaml
metadata:
  name: cyberether-blueprint-plugin
  version: 0.1.0
  minimumJetstreamVersion: 1.20.0

targets:
  - path: targets/macos-arm64-cpu/cyberether_blueprint_plugin.dylib
    system: macos
    device: cpu
    arch: arm64

examples:
  - path: examples/blueprint_gain.yml
```

| Field | Purpose |
|---|---|
| `metadata.name` | Unique plugin bundle name. |
| `metadata.version` | Plugin version in `x.y.z` format. |
| `metadata.minimumJetstreamVersion` | Minimum CyberEther or Jetstream version required (`x.y.z`). |
| `targets[].path` | Path to the shared library inside the archive. |
| `targets[].system` | Target operating system (`macos`, `linux`, `windows`, and so on). |
| `targets[].device` | Required device backend (`cpu`, `cuda`, `metal`, `vulkan`, or `webgpu`). |
| `targets[].arch` | Target CPU architecture (`arm64`, `x86_64`, and so on). |
| `examples[].path` | Path to an optional bundled example flowgraph. |

Version fields must contain exactly three decimal components between `0` and `255` with no prerelease or build suffixes.

> [!IMPORTANT]
> Treat `targets[].device` as a compatibility discriminator for one shared-library variant, never as a list of backends compiled into one file. CyberEther loads **every** target matching the current OS, architecture, and enabled backends. Listing the same library under multiple device labels causes CyberEther to load it twice and collide on static registrations.
>
> Instead, compile one library per device backend (keeping shared block registrations in each variant while ensuring module provider keys never overlap across compatible variants), package each into a single-target `.cep`, and combine them with `merger.py`.

## Managing Native Dependencies

Because users will drop your `.cep` file onto machines you do not control, aim for zero external setup:

- Depend only on ubiquitous system libraries (such as the C and C++ standard libraries).
- **Statically link** all third-party libraries directly into your plugin shared library whenever possible.
- If a dynamic dependency cannot be avoided, document every required library, its version range, and the exact package name for every supported OS. Missing shared libraries are painful for users to debug, so treat undocumented runtime dependencies as a packaging bug.

## Building Standalone

Compile the blueprint directly from its own directory:

```bash
cd examples/plugins/blueprint
meson setup build -Ddevices=cpu
meson compile -C build
```

On Linux, this produces the ready-to-install bundle at `build/cyberether_blueprint_plugin.cep`. Thanks to `subprojects/cyberether.wrap`, Meson automatically fetches CyberEther as a fallback when no system installation is found. The raw shared library in `build/` is only an intermediate artifact, while the `.cep` file is what you distribute.

### Building for the Browser (WebAssembly)

To build a browser-compatible plugin, cross-compile with Emscripten and CyberEther's cross file so your side module matches the host's threading, exception, SIMD, and LTO flags:

```bash
meson setup build-wasm examples/plugins/blueprint \
  --cross-file meson/crosscompile/emscripten.ini \
  -Dbuildtype=release \
  -Ddevices=cpu \
  -Dtests=false
meson compile -C build-wasm cyberether_blueprint_plugin_cep
```

## Bundling and Merging for Release

### Creating a Bundle With `bundler.py`

Run `tools/bundler.py` from your plugin directory to assemble a `.cep` file manually:

```bash
./tools/bundler.py \
  --output build/cyberether_blueprint_plugin.cep \
  --name cyberether-blueprint-plugin \
  --version 0.1.0 \
  --minimum-jetstream-version 1.20.0 \
  --target path=build/cyberether_blueprint_plugin.dylib,system=macos,device=cpu,arch=arm64 \
  --example examples/blueprint_gain.yml
```

Repeat `--target` for distinct library variants and `--example` for additional flowgraphs. Each `(system, device, arch)` tuple must be unique, and the same source library cannot be reused under multiple device labels for the same system and architecture.

### Combining Multi-Platform Builds With `merger.py`

When release pipelines build each platform on its own runner, combine the resulting single-target `.cep` files into one universal bundle using `tools/merger.py`:

```bash
./tools/merger.py \
  --output build/cyberether_blueprint_plugin.cep \
  build/macos-arm64.cep \
  build/linux-x86_64.cep
```

Before writing, `merger.py` validates that:

- Every input bundle shares identical metadata (`name`, `version`, and `minimumJetstreamVersion`).
- Each `(system, device, arch)` combination appears in at most one input bundle.
- Duplicate example paths have identical file contents (and deduplicates them cleanly).

Output is written atomically so an interrupted merge never leaves a corrupt `.cep` file behind.

## How CyberEther Loads a Bundle

When a user installs or loads your plugin, CyberEther executes nine steps:

1. Copies the `.cep` archive into the plugin cache.
2. Extracts the `tar.gz` contents into a cache folder.
3. Parses and validates `manifest.yml`.
4. Selects every target matching the current OS, architecture, and compiled device backends.
5. Opens each compatible shared library.
6. Resolves the exported plugin ABI symbol for each library.
7. Verifies ABI magic, record size, and ABI version.
8. Drains static block and module registrations into the global registry.
9. Registers any bundled examples from `examples[].path`.

To test installing and reloading your finished `.cep` bundle inside CyberEther, follow [Installing Plugins](/docs/installing-plugins).
