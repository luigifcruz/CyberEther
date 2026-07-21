---
title: Blocks And Modules
description: How to write native processing blocks and their compute modules.
order: 79
category: Development
---

Everything a user drops into a flowgraph is a block, and every block that computes something owns one or more modules. This page explains the split, the lifecycle of each half, the contract with the scheduler, and how implementations are registered and dispatched. It is the companion to [Metadata](/docs/metadata), which covers how blocks communicate, and [Creating Plugins](/docs/plugins), which covers packaging blocks into loadable libraries. If you want custom processing without writing C++ at all, see the [Python Block](/docs/python-block).

## The Split

A **block** is the user-facing graph node. It carries the identity (type, domain, title, description), the configuration fields, the declared inputs and outputs, and the orchestration logic that decides which modules to instantiate. A **module** is a compute kernel implemented for a specific device, runtime, and provider. The block is what the user sees and serializes, the module is what the scheduler runs.

The split exists because one logical operation often needs several implementations or several stages. A block can pick a different module depending on its configuration, chain multiple modules internally and wire them together, or own no module at all when it only orchestrates. The file reader block, for example, defines the user-facing configuration and metrics while a file reader module does the I/O.

Both halves follow the same structural pattern: a `Config` struct that describes and serializes the parameters, and an `Impl` struct that implements lifecycle hooks. The `DynamicConfig<Config>` mixin gives each a staged copy (the applied configuration) and a candidate copy (the pending edit), which is how reconfiguration validates changes before committing them.

## File Layout

Every shipped block follows the same layout, split between the public headers and the implementation tree. Using the AGC block as the example:

```
include/jetstream/domains/dsp/agc/
├── block.hh                    Block Config: type, domain, description, fields
└── module.hh                   Module Config: the parameters the kernel takes

src/domains/dsp/agc/
├── block_impl.cc               Block Impl: define(), create(), registration
├── module_impl.hh              Module Impl base shared by the device variants
├── module_impl.cc              Device-agnostic module logic
├── module_impl_native_cpu.cc   CPU kernel and its JST_REGISTER_MODULE
├── block_tests.cc              Block lifecycle and behavior tests
├── module_tests.cc             Module-level numeric tests
└── meson.build                 Adds the sources and tests to the build
```

The headers hold only the `Config` structs, since that is all other code needs to create the block programmatically. Everything else, including the `Impl` structs and the registrations, stays in the implementation files. Adding a backend means adding one file next to the CPU variant, for example `module_impl_native_cuda.cc`, with its own registration under `DeviceType::CUDA`. The suffix names the runtime and the device, which is how the Python runtime's `module_impl_python.cc` fits the same scheme. New blocks should copy this shape, and the plugin blueprint described in [Creating Plugins](/docs/plugins) mirrors it for out-of-tree development.

## Block Lifecycle

A block implementation derives from `Block::Impl` and overrides some of four hooks, all optional:

| Hook | Purpose |
|---|---|
| `validate()` | Reject bad configurations before anything is built. Returning `RECREATE` is also accepted here. |
| `configure()` | Derive internal state from the validated configuration. |
| `define()` | Declare the interface: inputs, outputs, configuration fields, and metrics. |
| `create()` | Build the modules and wire them to the block ports. |

There is no block-level destroy hook in practice. Destruction tears down the child modules automatically in reverse creation order, so anything that needs cleanup belongs in a module's own `destroy()`.

Creation runs `validate`, `configure`, and `define` in that order, then checks that every declared input is connected and resolved, runs `create()`, and finally checks that every declared output was produced. The result is one of three user-visible states:

- **Created.** Everything succeeded and the block participates in compute.
- **Incomplete.** The block is valid but cannot run yet. This happens automatically when a declared input is unconnected or its upstream is not producing, and deliberately when `create()` returns `Result::INCOMPLETE`. Incomplete is not an error, it is a waiting state.
- **Errored.** A hook failed, an undeclared input arrived, an expected output was missing, or a module later failed during compute. The failure message is kept as the block diagnostic. When a module fails at runtime, its block becomes errored and downstream blocks are recreated into the incomplete state.

The deliberate `INCOMPLETE` return is the gating pattern: a block whose `create()` needs a value that arrives later, such as an environment key delivered by a server connection, returns incomplete and is automatically destroyed and recreated when a new environment key becomes visible. The full pattern, with example code, is in [Flowgraph Environment](/docs/metadata#flowgraph-environment).

Configuration edits go through `reconfigure`, which validates the candidate configuration and applies it. When a change cannot be applied in place, for example a buffer size that shaped an allocation, return `Result::RECREATE` and the flowgraph destroys and recreates the block along with everything downstream of it.

## Defining The Block

The `Config` struct describes the block through macros and serializes its fields:

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
```

- `JST_BLOCK_TYPE` sets the string used in flowgraph files and for registry lookup.
- `JST_BLOCK_DOMAIN` places the block in a picker category. The shipped blocks use `Core`, `DSP`, `IO`, and `Visualization`.
- `JST_BLOCK_PARAMS` generates `serialize`, `deserialize`, and `hash` for the listed fields. The hash is what change detection uses, so every field that affects behavior belongs in the list. Field types follow the parser type system described in [Type Conversion](/docs/python-block#type-conversion).
- `JST_BLOCK_NODE_SIZE` picks the default node footprint in the editor from `XS`, `S`, `M`, `L`, and `XL`. Blocks that omit it default to `S`.
- `JST_BLOCK_DESCRIPTION` sets the title shown on the node, the one-line summary shown in the picker, and the long text shown in the documentation pane.

The `Impl` declares the interface in `define()` and builds modules in `create()`:

```cpp
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

The building vocabulary inside `create()`:

- `moduleCreate(name, config, inputs)` instantiates a module by its config type, resolved against the block's device, runtime, and provider, and feeds it the given tensor links.
- `moduleExposeOutput(blockPort, {module, modulePort})` publishes a module output as a block output. Every output declared in `define()` must be exposed, or creation fails.
- `moduleGetOutput({module, modulePort})` fetches an intermediate output to feed into the next module when chaining.
- `defineInterfaceConfig` and `defineInterfaceMetric` add editable fields and published values to the node UI. Metrics are covered in [Block Metrics](/docs/metadata#block-metrics).

Blocks may also access `environment()`, `view()`, `scheduler()`, and `render()` for the surrounding machinery.

## Module Lifecycle

A module implementation derives from `Module::Impl`, a runtime context that carries the compute hooks for its runtime, and `Scheduler::Context` for the scheduling hooks:

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
        JST_CHECK(output.create(device(), input.dtype(), {input.size()}));
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

The module lifecycle mirrors the block with `validate`, `define`, `create`, `destroy`, and `reconfigure`. Outputs are real tensors owned by the module and published with `produced()`. The `reconfigure` hook can apply cheap changes in place or return `RECREATE` to force a rebuild.

### Reconfiguring In Place

An in-place reconfigure reads the pending edit from `candidate()`, decides which changes it can absorb, and commits them by assigning the live fields. The RRC filter module is the reference example: a change to the tap count reshapes an allocation, so it recreates, while every other parameter only requires regenerating coefficients into the existing buffer:

```cpp
Result RrcFilterImpl::reconfigure() {
    const auto& config = *candidate();

    if (config.taps != taps) {
        return Result::RECREATE;
    }

    symbolRate = config.symbolRate;
    sampleRate = config.sampleRate;
    rollOff = config.rollOff;

    JST_CHECK(generateCoefficients());

    return Result::SUCCESS;
}
```

Two details make this work. The implementation inherits its config fields through `DynamicConfig`, so the members being assigned are the applied configuration itself, and `candidate()` holds the not-yet-applied edit to compare against. And returning `SUCCESS` means the module took full responsibility for the change, so everything the changed fields influence, such as the coefficient table here, must be refreshed before returning. When in doubt about whether an in-place path covers a field, return `RECREATE` and let the flowgraph rebuild.

### Compute Contract

The compute hooks come from the runtime context, for example `computeInitialize`, `computeSubmit`, and `computeDeinitialize` on `NativeCpuRuntimeContext`. The scheduler executes modules in topological order once per cycle, and what `computeSubmit` returns matters:

| Return | Meaning |
|---|---|
| `SUCCESS` | Normal completion. |
| `RELOAD` | Accepted as successful progress by the runtime. |
| `SKIP` | Nothing to do this cycle. The module is skipped and the skip propagates to every downstream module for the rest of the cycle. |
| `YIELD` | Abort the rest of the cycle quietly. Used when a precondition disappeared mid-cycle. |
| `TIMEOUT` | Abort the rest of the cycle quietly after a bounded wait or unavailable resource. |
| `ERROR` | The module failed. Its block becomes errored with the last log message as the diagnostic, and downstream blocks are recreated as incomplete. The rest of the flowgraph keeps running. |

Two rules govern the whole design:

- **Compute must never block.** The scheduler is synchronous, so a stalled `computeSubmit` stalls every block in the flowgraph. Anything that waits on the outside world belongs in a background thread that `computeSubmit` drains.
- **Sources pace the graph.** A module with no inputs is a source, and before each cycle the scheduler polls every source's `hasPendingCompute()` from `Scheduler::Context`. The default returns immediately, which makes the graph free-running. A real ingest source blocks there until data is available, which is what gives a flowgraph its natural rate. Note that the scheduler waits for all sources, so a slow source paces everything.

### Module Taints

Taints are opt-in capability and behavior contracts declared from the module's `define()` hook with `defineTaint`. The framework uses them while validating inputs, constructing the runtime plan, scheduling compute, and presenting surfaces.

| Taint | Module contract | Framework behavior |
|---|---|---|
| `CLEAN` | No special capabilities or scheduling guarantees. | Applies the default validation and executes the module every eligible cycle. It is represented by zero, so it does not need to be declared. |
| `IN_PLACE` | The module may overwrite storage belonging to one or more input tensors. | Static settlement is conservatively disabled for a graph containing the module because retained or aliased static buffers could otherwise be mutated. |
| `DISCONTIGUOUS` | The implementation correctly handles input tensor shapes, strides, and offsets instead of assuming tightly packed storage. | Allows non-contiguous input tensors during module creation. Without it, non-contiguous inputs are rejected. |
| `SURFACE` | The module owns a renderable surface and implements the presentation lifecycle. | Initializes the presentation context and includes the module in scheduler `present()` submissions. Compute settlement does not stop presentation. |
| `BROWSER_MAIN_THREAD` | Browser-side creation and destruction must execute on the browser's main runtime thread. | Proxies the module's create and destroy hooks to the main thread in browser builds. It has no effect on native builds. |
| `CROSS_DEVICE` | The implementation can directly consume input tensors whose device differs from the module's device. | Allows cross-device input during module creation. Without it, device mismatch is rejected and the block must provide an explicit bridge. |
| `THROTTLED` | Compute should run at the scheduler's slower fixed cadence rather than every loop cycle. | Defers the module until its deadline. While it is not due, the module is added to the cycle's skipped set and its downstream dependents are skipped. |
| `STATIC_OUTPUT` | After one successful compute, every output remains valid and unchanged for the current scheduler state epoch, and omitting later compute calls has no observable side effects. | Treats the module as an explicit static root. After successful materialization, the scheduler omits it without adding it to the skipped set and stops polling it as a source. |
| `STATELESS` | For fixed configuration and logical input values, compute has the same outputs and return behavior and has no externally observable side effects. | Allows staticness to propagate from static inputs. A stateless module settles only when every input has a known static producer. The `STATELESS` trait alone does not make a zero-input module a static root. |

Multiple traits can be combined in one declaration:

```cpp
Result MultiplyImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS |
                          Module::Taint::STATELESS));
    // Define ports...
    return Result::SUCCESS;
}
```

Traits belong to the concrete module implementation. A declaration in a shared implementation base applies to every CPU, CUDA, or other backend implementation that inherits that `define()` hook, so the contract must hold for all of them.

#### Static Settlement

The `STATIC_OUTPUT` and `STATELESS` traits avoid redundant compute without changing the meaning of `SKIP`. Staticness is derived from explicit static roots through stateless transformations. For example, a static Window can feed stateless Invert and Reshape modules before the result is combined with a live signal by a stateless Multiply and passed to a stateless FFT. Window, Invert, and Reshape settle after their first successful cycle. Multiply remains dynamic because one input comes from the live signal, and FFT remains dynamic because its Multiply input is dynamic.

The scheduler applies these rules:

- The `STATIC_OUTPUT` trait starts a static branch regardless of whether the module has inputs.
- The `STATELESS` trait propagates staticness only when the module has inputs and every input producer is known and static.
- Missing producers, producerless inputs, live producers, and unannotated modules stop propagation. Resolved cross-block links retain their module producer and can propagate staticness.
- A static candidate settles only after its runtime segment completes successfully and the module was not skipped.
- The `SKIP`, `YIELD`, `TIMEOUT`, and failure results do not settle a module. The scheduler retries it on a later cycle.
- Settled modules are omitted from runtime submission and source polling, but are never added to `skippedModules`. Their retained output remains available to dynamic consumers and across runtime or device segment boundaries.
- Timing presentation reports zero current compute cost for settled modules. Before invalidation, the scheduler restores its pre-settlement timing snapshot. A runtime rebuild that follows initializes fresh timing.
- If every module is settled, the scheduler idles instead of spinning at full speed.

Settlement lasts for a scheduler state epoch:

| Event | Settlement behavior |
|---|---|
| Successful compute followed by another cycle | Preserved because settled modules are omitted. |
| Scheduler `stop()` followed by `start()` | Preserved because runtime and module state remain initialized. |
| Synchronized state mutation | Cleared so static roots and descendants can refresh. |
| Module add, remove, or scheduler reload | Cleared while the runtime plan is rebuilt. |
| Block recreation or configuration change requiring recreation | Cleared through module removal and re-addition. |
| Presentation of a settled `SURFACE` module | Presentation continues while only compute submission is settled. |

The `STATIC_OUTPUT` contract is stronger than "usually does not change." The annotation is incorrect if output can change because of time, randomness, mutable external memory, input changes, or retained history. A static module may use internal caches, compiled kernels, plans, or scratch resources, but those implementation details cannot make its observable output vary after materialization.

Static-root examples include Window, FilterTaps, and OnesTensor. These modules derive their output from configuration and retain it after the first successful compute.

#### Stateless Contract

A module may declare `STATELESS` only when all of the following are true:

- Output values and compute return behavior depend only on the current configuration and input values.
- Compute does not depend on previous invocations, wall-clock time, random state, files, devices, queues, environment state, or externally mutable memory.
- Compute does not publish metrics, write files, render, perform I/O, mutate inputs, or cause other observable side effects.
- Repeating compute with the same inputs is semantically equivalent even if the implementation reuses internal plans, kernels, or scratch buffers.

Stateless transform examples include Invert, Reshape, Multiply, FFT, Duplicate, Cast, MultiplyConstant, Pad, Unpad, Arithmetic, Range, AGC, Amplitude, and Fold. They can still execute every cycle when connected to live input. The trait only makes them eligible to inherit staticness.

Examples that must not declare `STATELESS` include SignalGenerator because it advances sample state and may use randomness, FileReader because it advances an external stream, RrcFilter and OverlapAdd because they retain history, Throttle because it depends on time, Squelch because it publishes metrics and controls downstream scheduling, and Python because user code may perform arbitrary stateful operations.

The `STATELESS` and `IN_PLACE` traits are conceptually incompatible because mutating an input is an observable side effect. The scheduler currently disables static settlement whenever `IN_PLACE` appears in the graph because tensor views can alias storage across more than one module.

Most importantly, settlement is not `SKIP`. A `SKIP` result means output is unavailable for the current cycle and therefore propagates to downstream modules. Settlement means a previously computed output is still valid, so downstream modules continue normally.

## Registration And Dispatch

Registration is static. Placing the macros at namespace scope in a translation unit queues the registration, and both the main binary and loaded plugins drain into the same registry:

```cpp
JST_REGISTER_BLOCK(GainBlock, {"gain"});
JST_REGISTER_MODULE(GainModuleNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");
```

A block registration lists every module type its implementation may create. Requirements used only by a configuration-dependent path are marked conditional, for example `{"agc", true}`. Blocks with no child modules use `JST_REGISTER_BLOCK_NO_MODULES`. The picker advertises targets that provide every unconditional requirement, while module resolution during block creation remains authoritative.

A block is looked up by its type string alone. A module is looked up by four keys: its config type string plus the device, runtime, and provider of the block that is creating it. This is the dispatch mechanism, and it is why one block works across backends: registering a second module under the same type with `DeviceType::CUDA` makes the same block work on a CUDA device with no block changes.

The provider string distinguishes alternative implementations of the same module on the same device and runtime. Most modules register as `generic`, which is also the default a block is created with. A plugin can register a specialized provider, for example one backed by a vendor library, and blocks created with that provider pick it up.

Practical hints:

- **Keep blocks thin.** Configuration handling, interface definition, and wiring belong in the block. Anything that touches samples belongs in a module, where it can be reimplemented per device without touching the block.
- **Prefer `RECREATE` over clever in-place reconfiguration.** The flowgraph already knows how to tear down and rebuild a block and its downstream correctly. In-place reconfiguration is an optimization for cheap changes, not the default.
- **Return `INCOMPLETE` instead of faking readiness.** A block that cannot run yet should say so and let the retry machinery bring it up when its dependency appears. Blocks that pretend to be created and internally idle are invisible to the user and to the tooling.
- **Test the lifecycle, not just the math.** The block test suites under `src/domains` cover creation, incomplete transitions, reconfiguration, and recreation alongside numeric results, and new blocks should do the same.
