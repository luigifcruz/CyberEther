---
title: Blocks And Modules
description: How native blocks and compute modules split responsibilities, organize files, and register for dispatch.
order: 320
category: Development
---

Every node a user drops into a CyberEther flowgraph is a **block**, and every block that processes data owns one or more **modules**. This guide introduces how those two halves split responsibilities, how built-in files are laid out, how a minimal block-and-module pair fits together, and how registrations dispatch across devices.

- For block lifecycle hooks, states, validation rules, and UI control descriptors, see [Block Lifecycle And UI](/docs/block-lifecycle).
- For module hooks, `computeSubmit()` codes, taints, static settlement, and interactive surface input, see [Module Contract](/docs/module-contract).
- Read this alongside [Metadata](/docs/metadata) (environment keys, tensor attributes, and metrics) and [Creating Plugins](/docs/plugins) (shipping blocks outside the main tree). If you want custom processing without writing C++, see the [Python Block](/docs/python-block).

## The Block and Module Split

```text
  Block (User-Facing Node)
   ├── Identity, ports, UI controls, metrics, YAML config
   └── Orchestrates one or more Modules
         └── Module (Device Kernel)
              └── Implements compute (or rendering) for one Device + Runtime + Provider
```

- A **block** is the graph node users interact with and serialize. It defines the type string, domain, title, description, editable fields, declared ports, and orchestration logic.
- A **module** is a concrete compute or surface kernel built for a specific device, runtime, and provider. The block is what users see, while the module is what the scheduler executes.

Keeping the two separate lets a single logical block select different kernels based on its configuration, wire multiple internal modules together, or own no modules at all when it only orchestrates. In the File Reader, for example, the block exposes settings and progress metrics while the module handles file I/O.

Both halves share the same configuration pattern: a `Config` struct describing serializable fields, and an `Impl` struct implementing lifecycle hooks. Inheriting `DynamicConfig<Config>` gives each half both a staged copy (the currently applied configuration) and a `candidate()` copy (the pending edit), allowing validation to inspect changes before committing them.

## Standard File Layout

Built-in blocks follow one consistent layout split between public headers and implementation files. Here is the **AGC** block as a template:

```text
include/jetstream/domains/dsp/agc/
├── block.hh                    Block Config: type, domain, description, fields
└── module.hh                   Module Config: parameters passed to the kernel

src/domains/dsp/agc/
├── block_impl.cc               Block Impl: define(), create(), registration
├── module_impl.hh              Shared Module Impl base across device variants
├── module_impl.cc              Device-agnostic module logic
├── module_impl_native_cpu.cc   CPU kernel and JST_REGISTER_MODULE
├── block_tests.cc              Block lifecycle and behavior tests
├── module_tests.cc             Module-level numeric tests
└── meson.build                 Registers sources and tests with Meson
```

Public headers contain only `Config` structs so external code can construct blocks programmatically without pulling in implementation details. Everything else stays under `src/`.

Adding a new device backend means adding a sibling file, such as `module_impl_native_cuda.cc`, registered under `DeviceType::CUDA`. The filename suffix identifies both runtime and device (for example, `module_impl_python.cc` for the Python runtime). Out-of-tree plugins mirror this exact layout (see [Creating Plugins](/docs/plugins)).

## Minimal Gain Walkthrough

A minimal processing node consists of a block config and implementation paired with a module config and device kernel.

### Block Side

Describe the block's metadata and serializable fields in `block.hh`, then implement `define()` and `create()` in `block_impl.cc`:

```cpp
struct GainConfig : Block::Config {
    F32 gain = 1.0f;

    JST_BLOCK_TYPE(gain);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(gain);
    JST_BLOCK_NODE_SIZE(S);
    JST_BLOCK_DESCRIPTION("Gain",
                          "Multiplies the signal by a constant.",
                          "Longer text shown in the documentation pane.");
};

struct GainBlock : Block::Impl, DynamicConfig<GainConfig> {
    Result define() override {
        JST_CHECK(defineInterfaceInput("signal", "Signal", "Input tensor."));
        JST_CHECK(defineInterfaceOutput("signal", "Signal", "Scaled tensor."));
        return Result::SUCCESS;
    }

    Result create() override {
        auto moduleConfig = std::make_shared<GainModuleConfig>();
        moduleConfig->gain = gain;
        JST_CHECK(moduleCreate("gain", moduleConfig, {{"signal", inputs().at("signal")}}));
        JST_CHECK(moduleExposeOutput("signal", {"gain", "signal"}));
        return Result::SUCCESS;
    }
};

JST_REGISTER_BLOCK(GainBlock, {"gain"});
```

### Module Side

Describe the kernel parameters in `module.hh`, allocate and publish the output tensor in `create()`, and run the kernel in `computeSubmit()`:

```cpp
struct GainModuleConfig : Module::Config {
    F32 gain = 1.0f;

    JST_MODULE_TYPE(gain);
    JST_MODULE_PARAMS(gain);
};

struct GainModuleNativeCpu : Module::Impl,
                             DynamicConfig<GainModuleConfig>,
                             NativeCpuRuntimeContext,
                             Scheduler::Context {
    Result define() override {
        JST_CHECK(defineInterfaceInput("signal"));
        return defineInterfaceOutput("signal");
    }

    Result create() override {
        const auto& input = inputs().at("signal").tensor;
        JST_CHECK(output.create(device(), input.dtype(), input.shape()));
        JST_CHECK(output.propagateAttributes(input));
        outputs()["signal"].produced(name(), "signal", output);
        return Result::SUCCESS;
    }

    Result computeSubmit() override {
        // Process inputs into the output tensor here.
        return Result::SUCCESS;
    }

    Result reconfigure() override {
        return Result::RECREATE;
    }

    Tensor output;
};

JST_REGISTER_MODULE(GainModuleNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");
```

Outputs are real `Tensor` objects owned by the module and published through `produced()`.

## Registration And Dispatch

Registrations happen statically at namespace scope, draining into the same global registry from both the main binary and loaded plugins:

```cpp
JST_REGISTER_BLOCK(GainBlock, {"gain"});
JST_REGISTER_MODULE(GainModuleNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");
```

- List every module type a block may instantiate in `JST_REGISTER_BLOCK`. Mark configuration-dependent modules as conditional, such as `{"agc", true}`. Use `JST_REGISTER_BLOCK_NO_MODULES` for blocks with no child modules. The block picker advertises targets satisfying all unconditional requirements, while creation-time module lookup remains authoritative.
- Blocks look up by type string alone. Modules look up by four keys: **module type**, **device**, **runtime**, and **provider**. Adding a `DeviceType::CUDA` module under the same type string immediately enables CUDA support for the existing block with zero block changes.
- The **provider** string distinguishes competing implementations on the same device and runtime. Most modules register under `"generic"` (the default), while plugins can register specialized vendor-backed providers.

## Practical Guidelines

- **Keep blocks thin.** Keep config handling, UI definitions, and wiring in the block, and put all sample processing in modules so new device backends never touch block code.
- **Default to `RECREATE`.** The flowgraph already tears down and rebuilds subgraphs cleanly. Reserve in-place reconfiguration for genuinely cheap updates.
- **Return `INCOMPLETE` instead of idling.** A block that lacks prerequisites should report `INCOMPLETE` so retry logic brings it online automatically rather than hiding in an idle "created" state.
- **Test the full lifecycle.** Mirror the test suites under `src/domains`, covering creation, incomplete transitions, reconfiguration, recreation, and numeric accuracy.
