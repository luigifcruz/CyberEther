---
title: Module Contract
description: Module lifecycle hooks, computeSubmit return codes, scheduler rules, taints, static settlement, and surface input.
order: 322
category: Development
parent: blocks-and-modules
---

Modules are the concrete compute and surface kernels executed by the scheduler for a specific device, runtime, and provider. This page defines the full module lifecycle, compute contract, taint flags, static settlement rules, and interactive surface input API.

## Module Lifecycle

A module implementation derives from `Module::Impl`, a runtime context providing compute hooks (such as `NativeCpuRuntimeContext`), and `Scheduler::Context`:

| Hook | Responsibility |
|---|---|
| `validate()` | Validate candidate config semantics and implementation-specific input compatibility without side effects. |
| `define()` | Declare module taints and input/output ports. |
| `create()` | Allocate buffers, initialize external state, and publish outputs via `produced()`. |
| `destroy()` | Release resources and external state created by the module. |
| `reconfigure()` | Apply a validated edit in place or return `Result::RECREATE`. |

### Creation Sequence

Module creation validates before defining ports and running framework input checks:

```text
Module::create
├── Store provided inputs
├── Deserialize candidate configuration
├── validate()
├── Commit candidate configuration
├── define() taints and ports
├── Framework input checks
│   ├── Every declared input is present
│   ├── Device matches (unless CROSS_DEVICE)
│   ├── Tensor shape is initialized and non-empty
│   └── Storage is contiguous (unless DISCONTIGUOUS)
├── create()
└── Verify every declared output was produced
```

### Validation Ownership

| Layer | What It Checks |
|---|---|
| Shared `validate()` | Config ranges and relationships, signal axes, rank, dimensions, derived sizes, and other backend-independent rules. |
| Provider `validate()` | Supported dtypes, backend limits, representable allocation sizes, and device capabilities. Call the shared `validate()` first. |
| Framework checks | Missing inputs, device mismatches, invalid or empty tensors, and contiguity (relaxed only by matching taints). |
| `create()` | No deterministic validation. Consumes already-validated values to allocate resources and publish outputs. |

Because `validate()` runs before the framework's input checks, it must tolerate a missing, uninitialized, or empty input by returning `Result::SUCCESS` for the input check so the framework emits the standard lifecycle error. Still validate candidate config fields even when inputs are absent, and always read pending values from `candidate()`. Validation may compute and cache bounded derived sizes for `create()`, though it must never allocate resources, open hardware, or trigger external side effects.

### Reconfiguring in Place

To absorb cheap edits without rebuilding buffers, compare `candidate()` against your applied fields, update the applied members, refresh any derived state, and return `Result::SUCCESS`. For example, the RRC filter recreates if `taps` changes (because buffer size changes) and regenerates coefficients in place for every other field:

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

Returning `Result::SUCCESS` signals that your module refreshed everything affected by the changed fields. When in doubt, return `Result::RECREATE` and let the flowgraph rebuild cleanly.

Before invoking `reconfigure()`, the framework snapshots the serialized applied configuration and attempts to roll it back if your hook fails (entering `ERRORED` if restoration itself fails). Rollback restores config fields only, so your implementation remains responsible for any runtime, hardware, or external side effects.

### Module-Originated Configuration Edits

Interactive surface modules can propose configuration edits without mutating applied fields directly. First, the owning block binds a module field to a block config field after `moduleCreate()`:

```cpp
JST_CHECK(moduleCreate("detector", detectorConfig, inputs));
JST_CHECK(moduleBindConfigEdit("detector", "threshold", "threshold"));
```

Both field names must exist in their respective serialized configurations. The module can then queue a patch in response to user interaction:

```cpp
Parser::Map edit;
edit["threshold"] = newThreshold;
JST_CHECK(requestConfigChange(edit));
```

Queued patches apply through standard block reconfiguration before the next compute cycle or export. Any unbound field rejects the entire patch, and repeated edits to the same field coalesce to the latest value. Use `configChangeEnabled(key)`, `configChangePending()`, and `configChangeResult()` to drive UI controls, keeping live drag previews separate from applied config and emitting requests only on user input (never in response to an accepted edit). The **Spectrum Analyzer** uses this pattern for `splitRatio`, previewing drags smoothly and persisting the final ratio in place without losing trace averaging or waterfall history.

## Compute Contract

Runtime contexts supply compute hooks (such as `computeInitialize`, `computeSubmit`, and `computeDeinitialize` on `NativeCpuRuntimeContext`). On each cycle, the scheduler executes eligible modules in topological order at most once per cycle whenever every input carries an unread frame.

### `computeSubmit()` Return Codes

| Return Value | Meaning |
|---|---|
| `SUCCESS` | Normal completion. Downstream consumers receive a fresh output frame. |
| `RELOAD` | Accepted as successful progress by the runtime. |
| `SKIP` | Nothing to emit this cycle (for example, a decimator accumulating input). Emits an empty frame causing downstream modules to drop their held frames without running. Writing partial output before returning `SKIP` is safe because empty frames are never read. |
| `YIELD` | Not ready yet. Inputs stay unread, dependent modules wait, and the scheduler retries on a later cycle while the rest of the graph continues. |
| `TIMEOUT` | Equivalent to `YIELD`. Sources return `TIMEOUT` from `hasPendingCompute()` when no full batch is ready. |
| `ERROR` | Runtime failure. Marks the owning block **Errored** (with the last log line as diagnostic) and recreates downstream blocks as **Incomplete**, while unrelated branches keep running. |

### Four Rules of the Scheduler

1. **Never block on external timing.** All branches share one compute thread. Hardware or network sources (radios, sockets, audio) must read on a background thread into a ring buffer drained by `computeSubmit()` so late reads never drop data. File I/O and transmitting buffers already in memory run inline with bounded timeouts so slow sinks naturally pace their chain.
2. **Sources answer immediately.** A zero-input module is a source. Every cycle, the scheduler polls `hasPendingCompute()` on `Scheduler::Context`, which must return immediately with `SUCCESS` or `TIMEOUT` (the default returns `SUCCESS`, making a source free-running). Background ingest sources call `Scheduler::Context::NotifyPendingCompute()` after pushing a batch to wake an idle scheduler instantly.
3. **Frames stay strictly aligned.**
   - Every module consumes exactly one frame per input per step (by computing, dropping on an upstream skip, or dropping while throttled), and multi-input modules only fire when all inputs derive from the same source frames.
   - Producers reuse one output buffer and wait until every consumer finishes reading it (including view modules and attribute-inheriting or derived-attribute modules so held frames never observe newer samples or metadata).
   - Joining sources running at different rates paces the faster one until its buffer overflows, so resample or share a clock before joining.
   - Call `propagateAttributes()` and `setDerivedAttribute()` in `create()`, never inside `computeSubmit()`, because the scheduler inspects attribute links at graph build time.
4. **Shapes stay fixed.** Output shapes are published in `create()` and remain constant until a `RECREATE`.

## Module Taints

Taints are opt-in capability flags declared inside `define()` with `defineTaint()`. They guide input validation, scheduling, settlement, and presentation.

| Taint | Module Guarantee | Framework Behavior |
|---|---|---|
| `CLEAN` | Default behavior (value `0`, no need to declare). | Standard validation that executes every eligible cycle. |
| `IN_PLACE` | May overwrite input tensor storage. Must never overwrite memory another module is still reading. | Conservatively disables static settlement across the graph so aliased static buffers cannot be mutated. |
| `DISCONTIGUOUS` | Handles arbitrary tensor shapes, strides, and offsets safely. | Permits non-contiguous input tensors at creation instead of rejecting them. |
| `SURFACE` | Owns a renderable surface and implements presentation hooks. | Initializes presentation context and includes the module in `present()`. Compute settlement never pauses presentation. |
| `BROWSER_MAIN_THREAD` | Browser creation and destruction must run on the browser main thread. | Proxies `create()` and `destroy()` to the main thread in WebAssembly builds. No effect on native builds. |
| `CROSS_DEVICE` | Can directly consume inputs residing on another device. | Allows cross-device inputs at creation without an explicit bridge. |
| `THROTTLED` | Runs on the scheduler's slower fixed cadence instead of every cycle. | Executes at most once per throttle interval, consuming and dropping intermediate frames like a skip so producers keep flowing and consumers receive empty frames. |
| `STATIC_OUTPUT` | After one successful compute, outputs remain valid and unchanged for the current scheduler epoch, with no side effects from skipping later calls. | Treats the module as an explicit static root, omitting it from later compute and source polling once materialized. |
| `STATELESS` | Outputs and return codes depend purely on config and input values, with zero observable side effects. | Propagates staticness when every input comes from a known static producer. Does not make a zero-input module a static root on its own. |

Combine multiple flags with bitwise OR:

```cpp
Result MultiplyImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS |
                          Module::Taint::STATELESS));
    // Define ports...
    return Result::SUCCESS;
}
```

Taints declared in a shared base `define()` apply to every backend subclass (CPU, CUDA, and so on), so every backend must honor the contract.

### Static Settlement

Combining `STATIC_OUTPUT` and `STATELESS` eliminates redundant work without altering `SKIP` semantics. Imagine a static **Window** feeding stateless **Invert** and **Reshape**, which then feed a stateless **Multiply** alongside a live signal into a stateless **FFT**. Window, Invert, and Reshape compute once and settle. Multiply and FFT stay dynamic because one input is live.

How the scheduler settles modules:

- Every `STATIC_OUTPUT` module starts a static branch (with or without inputs).
- A `STATELESS` module inherits staticness only when it has inputs and every input producer is known and static (including across resolved block boundaries). Missing, live, or unannotated producers stop propagation.
- Settlement occurs only after a module's runtime segment completes normally without returning `SKIP`, `YIELD`, `TIMEOUT`, or `ERROR`.
- Settled modules skip runtime submission and source polling without entering `skippedModules`, carry no source frame numbers (so they never stall live joins), report zero current compute cost (restoring pre-settlement timing snapshots before invalidation), and keep their retained outputs available across device and runtime boundaries.
- When every module in a graph is settled, the cycle reports no progress and the instance sleeps instead of spinning.

| Event | Effect on Settlement |
|---|---|
| Normal compute cycle | Preserved. |
| Scheduler `stop()` then `start()` | Preserved because runtime and module state remain initialized. |
| Synchronized state mutation | Cleared so static branches refresh, and all unread frames drop simultaneously so branches restart aligned. |
| Module add or remove, or scheduler reload | Cleared while rebuilding the runtime plan. |
| Block recreation | Cleared via module removal and re-addition. |
| Presenting a settled `SURFACE` module | Presentation continues normally while compute stays settled. |

### Strict Rules for `STATIC_OUTPUT` and `STATELESS`

The `STATIC_OUTPUT` contract is stricter than "rarely changes." Never declare it if outputs vary with time, randomness, mutable external memory, input changes, or retained history (internal caches, compiled kernels, and scratch buffers are fine). Valid static roots include **Window**, **FilterTaps**, and **OnesTensor**.

Declare `STATELESS` only when all four conditions hold:

1. Outputs and return codes depend solely on current configuration and input values.
2. Compute never depends on prior calls, wall-clock time, randomness, files, devices, queues, environment keys, or external mutable memory.
3. Compute never publishes metrics, writes files, renders, performs I/O, mutates inputs, or produces other side effects.
4. Repeating compute on identical inputs is semantically identical (reusing internal plans or scratch buffers is fine).

Good `STATELESS` examples include **Invert**, **Reshape**, **Multiply**, **FFT**, **Duplicate**, **Cast**, **MultiplyConstant**, **Pad**, **Unpad**, **Arithmetic**, **Range**, **AGC**, **Amplitude**, and **Fold**.

Modules that must **never** declare `STATELESS` include **SignalGenerator** (advances phase or random state), **FileReader** (advances stream), **RrcFilter** and **OverlapAdd** (retain history), **Throttle** (depends on time), **Squelch** (publishes metrics and controls scheduling), and **Python** (runs arbitrary user code). Combining `STATELESS` and `IN_PLACE` is conceptually invalid because mutating inputs is a side effect.

> [!IMPORTANT]
> Settlement is not `SKIP`. Returning `SKIP` means no valid output exists for that frame, so downstream modules drop the frame too. Settlement means the previously computed output remains valid, so downstream modules keep running normally.

## Interactive Surface Input

Surface modules consume an ordered stream of `InputEvent` values (`std::variant<MouseEvent, KeyEvent, FocusEvent>`, declared in `jetstream/surface.hh`) via `surfaceConsumeInputEvents()`, alongside resize and placement events from `surfaceConsumeSurfaceEvents()`:

- **Mouse coordinates** are surface-normalized with `(0, 0)` at top left. Captured drags and releases may report coordinates outside `[0, 1]`.
- **Keyboard events** carry a Jetstream `KeyCode`, a `Press` or `Release` action, and a `repeat` flag (identifying physical keys rather than text input).
- **Modifiers** (Control, Shift, Alt, Super/Command) are snapshotted on both mouse and keyboard events, preserving native Control and Command behavior on macOS.
- **Focus lifecycle:** Clicking a surface grants keyboard focus, and moving the pointer outside does not clear it. Clicking elsewhere, focusing another widget, or hiding the surface releases held keys and emits `FocusEvent{false}` (hidden surfaces clean up at frame end even if skipped). Always cancel active captures and held-input state on focus loss.

Inspect variants with `std::get_if<KeyEvent>(&event)` or `std::get_if<MouseEvent>(&event)`. Mouse-only surfaces can call `SurfaceMouseEvent(event)`, which returns an optional mouse event and translates focus loss into `MouseEventType::Leave`. Existing zoom, pan, and cursor interactions work out of the box via `ProcessSurfaceInteraction()`. Interface components route input through `SurfaceView::Config::onInput` and `Module::Surface::pushInputEvent()`, consuming the queue once per presentation in order.
