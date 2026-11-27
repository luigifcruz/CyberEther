---
title: Creating Plugins
description: How to create external CyberEther plugins.
order: 360
category: Development
---

Plugins let you develop, build, and distribute custom CyberEther blocks outside the main source tree while using the exact same block, module, scheduler, runtime, and tensor APIs as built-in blocks. Each plugin ships as one `.cep` bundle per CyberEther build, containing a manifest, the compiled library, optional data files, and optional example flowgraphs.

## Starting From the Blueprint

The fastest way to start a new plugin is to copy the official blueprint included in the repository:

```text
examples/plugins/blueprint
```

After copying the directory, pick a plugin ID and rename the Meson project, include folder, source folder, block type, module type, and plugin metadata to match it.

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
│   └── gain.yml                         Bundled example flowgraph
├── src/
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
│   └── bundler.py                       Packages a .cep bundle
├── meson.build
└── meson_options.txt
```

## Plugin ID and Naming

Every plugin has a stable ID, such as `blueprint`. The ID uses 1 to 32 lowercase letters and digits and starts with a letter. CyberEther uses it to tell plugins apart, to name the plugin's install folder, and to record which plugins a flowgraph needs.

Everything a plugin registers must carry the ID as a prefix:

- **Block types** start with `<id>_`, for example `blueprint_gain`.
- **Implementations** either belong to one of the plugin's own blocks, or use a provider name that starts with `<id>_`. That second form lets a plugin add an implementation to a built-in block, for example a CUDA kernel for `fft` with the provider `blueprint_fast`.

CyberEther rejects a plugin that registers anything without the prefix.

## Registering Blocks and Modules

Plugins use the exact same registration macros as built-in blocks (see [Blocks And Modules](/docs/blocks-and-modules)). The blueprint implements a `blueprint_gain` block backed by a native CPU module supporting `F32` and `CF32` tensors:

```cpp
JST_REGISTER_BLOCK(BlueprintGainImpl, {"blueprint_gain"});

JST_REGISTER_MODULE(BlueprintGainImplNativeCpu,
                    DeviceType::CPU,
                    RuntimeType::NATIVE,
                    "generic");
```

When CyberEther opens your library, those static registrations drain straight into the global registry. No extra export or entry point is needed.

## CyberEther Versions and Builds

A plugin is built against one exact CyberEther version and loads only into that version. CyberEther also ships several release builds, and their libraries differ, so a plugin needs one library per build:

| Build | Platform |
|---|---|
| `macos-arm64` | macOS on Apple Silicon |
| `linux-x86_64` | Linux on x86_64 |
| `linux-x86_64-cuda` | Linux on x86_64 with CUDA |
| `linux-aarch64` | Linux on ARM64 |
| `linux-aarch64-cuda` | Linux on ARM64 with CUDA |
| `windows-x86_64` | Windows on x86_64 |

Each library contains every implementation whose backend that build supports. The CyberEther Meson dependency exports its build ID and its enabled backends, so the plugin build reads them and compiles the matching implementation files automatically:

```meson
cyberether_build_id = cyberether_dep.get_variable('build_id')
cyberether_backends = cyberether_dep.get_variable('backends').split(',')

if 'cuda' in cyberether_backends
    plugin_sources += files('module_impl_native_cuda.cc')
endif
```

The same build ID is available to C++ code as `JETSTREAM_BUILD_ID`.

## Anatomy of a `.cep` Bundle

A `.cep` file is a standard zip archive with a `.cep` extension. CyberEther finds everything by its location, so the manifest only describes the plugin:

```text
.
├── manifest.yml
├── lib/
│   ├── blueprint.so
│   └── libvendor.so.2
├── data/
│   └── taps.bin
└── examples/
    └── gain.yml
```

| Path | Purpose |
|---|---|
| `manifest.yml` | Plugin ID, title, version, build, and metadata. |
| `lib/<id>.<ext>` | The plugin library. The extension is `.dylib` on macOS, `.so` on Linux, and `.dll` on Windows. |
| `lib/` | Anything else native, such as bundled libraries, compiled kernels, or helper programs. |
| `data/` | Read-only files such as models, lookup tables, or firmware. |
| `examples/*.yml` | Example flowgraphs shown in the Start tab. |

Each bundle holds exactly one build. CyberEther rejects bundles with anything else at the top level.

Entries must be stored or deflated and not encrypted, and each path can appear only once. A bundle can be at most 512 MiB, expand to at most 2 GiB, and hold at most 10,000 entries, and `manifest.yml` can be at most 64 KiB. The blueprint's `tools/bundler.py` writes bundles that follow these rules.

The manifest is a flat YAML file:

```yaml
id: blueprint
title: Blueprint
version: 0.1.0
cyberether: 2.0.0
build: macos-arm64
summary: A minimal gain block to start new plugins from.
author: CyberEther
license: MIT
description: |-
  Longer text for the plugin detail page.
```

| Field | Purpose |
|---|---|
| `id` | Stable plugin ID. Required. |
| `title` | Display name. Required. |
| `version` | Plugin version, using letters, digits, `.`, `-`, or `+`. Required. |
| `cyberether` | Exact CyberEther version the plugin was built against. Required. |
| `build` | CyberEther build the library was compiled for. Required. |
| `summary` | One line shown in plugin lists. |
| `author` | Plugin author. |
| `license` | Plugin license. |
| `description` | Longer Markdown text for the plugin detail page. |
| `diagram` | Gallery card thumbnail, in the arrow syntax of a flowgraph file's `diagram`. |

Leave `diagram` out of bundles built for CyberEther versions released before the key existed, because they reject it as unknown.

CyberEther rejects manifests with unknown keys.

## Finding Bundled Files

Plugin code finds its own files through `Plugin::Path`, which returns the plugin's install folder with the same layout as the `.cep` bundle. It works from the moment the library opens, including inside static registrations:

```cpp
#include <jetstream/plugin.hh>

const auto root = Jetstream::Plugin::Path("blueprint");
const auto table = root / "data" / "taps.bin";
const auto kernel = root / "lib" / "gain.cubin";
```

## Managing Native Dependencies

Because users will install your `.cep` file on machines you do not control, aim for zero external setup:

- Depend only on ubiquitous system libraries (such as the C and C++ standard libraries) and on the libraries CyberEther itself already loads.
- **Statically link** third-party libraries into your plugin library whenever possible.
- Otherwise, **bundle** the library next to your plugin library in `lib/`. Your operating system loads it together with the plugin:
  - On Linux, set the `$ORIGIN` run path on the plugin library and on every bundled library that needs a sibling. Store each library under its real soname, since bundles cannot contain symbolic links.
  - On macOS, use `@rpath` install names with an `@loader_path` run path, then sign the libraries again after changing them.
  - On Windows, place the DLLs next to the plugin library.
- Two plugins that bundle different copies of the same library can clash. Rename your copy or link it statically.
- A library you only need sometimes, such as a vendor SDK for one backend, should not be linked at all. Open it from your implementation with the path from `Plugin::Path` when it is first used, so the rest of the plugin still loads without it.

## Building Standalone

Compile the blueprint directly from its own directory:

```bash
cd examples/plugins/blueprint
meson setup build
meson compile -C build
```

This produces the ready-to-install bundle at `build/blueprint-<version>-cyberether-<cyberether-version>-<build-id>.cep`. Thanks to `subprojects/cyberether.wrap`, Meson automatically fetches CyberEther as a fallback when no installation is found. The raw shared library in `build/` is only an intermediate artifact, while the `.cep` file is what you distribute.

## Creating a Bundle With `bundler.py`

Run `tools/bundler.py` from your plugin directory to assemble a `.cep` file manually:

```bash
./tools/bundler.py \
  --output build/blueprint.cep \
  --id blueprint \
  --title Blueprint \
  --version 0.1.0 \
  --cyberether 2.0.0 \
  --build macos-arm64 \
  --library build/blueprint.dylib \
  --summary "A minimal gain block to start new plugins from." \
  --example examples/gain.yml
```

| Option | Purpose |
|---|---|
| `--build <build-id>` | CyberEther build the library was compiled for. |
| `--library <path>` | Plugin library. It is renamed to `lib/<id>.<ext>` inside the bundle. |
| `--lib-file <path>` | Extra file or folder copied into `lib/`, such as a bundled library. |
| `--data <path>` | File or folder copied into `data/`. |
| `--example <path>` | Example flowgraph. |
| `--summary`, `--description`, `--author`, `--license` | Optional manifest metadata. |

Repeat `--lib-file`, `--data`, and `--example` as needed. To ship several builds, run the bundler once per build and publish each file.

## How CyberEther Installs and Loads a Bundle

When a user installs your plugin, CyberEther:

1. Extracts the bundle into its own plugin storage.
2. Reads and validates `manifest.yml`, including the exact CyberEther version and build.
3. Checks that `lib/<id>.<ext>` exists.

When it loads the plugin, CyberEther:

1. Opens the plugin library.
2. Drains its static block and module registrations into the global registry.
3. Checks that every registration carries the plugin ID prefix.
4. Publishes the bundled examples.

To test installing your finished `.cep` bundle inside CyberEther, follow [Installing Plugins](/docs/installing-plugins). While you iterate, load it with `--plugin` instead and use **Reload** after each rebuild.

## Publishing to the Gallery

Plugins published to the CyberEther gallery are rebuilt for every CyberEther release and build, and CyberEther offers those builds to users as updates, following the update policy each user picks in **Settings** > **Registry**. Users can link to your plugin with `https://cyberether.org/gallery/plugins/<id>`.
