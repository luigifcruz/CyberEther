---
title: Block Lifecycle And UI
description: Block hooks, lifecycle states, validation boundaries, reconfiguration, and UI control descriptors.
order: 321
category: Development
parent: blocks-and-modules
---

A CyberEther block manages the user-facing side of a node: its ports, editable controls, live metrics, YAML serialization, and child module orchestration. This page covers every block lifecycle hook and the `Parser::Map` interface descriptor system.

## Block Lifecycle Hooks

A block implementation inherits from `Block::Impl` and overrides any of five optional hooks:

| Hook | Responsibility |
|---|---|
| `validate()` | Check block-level candidate semantics affecting interface or orchestration. Do not duplicate child module checks. May return `RECREATE`. |
| `configure()` | Compute derived internal state from the validated configuration. |
| `define()` | Declare the candidate interface: inputs, outputs, config controls, and metrics. |
| `create()` | Instantiate child modules and wire them to block ports. |
| `reconfigure()` | Complete an in-place edit after all child modules have applied it. |

### Creation Sequence

```text
Block::create
├── Deserialize candidate configuration
├── define() candidate interface
├── validate() candidate semantics
├── Commit candidate configuration
├── configure() derived block state
├── Verify declared input links
├── create() child modules and wiring
└── Verify declared outputs
```

There is no block-level destroy hook. When a block is torn down, its child modules are destroyed automatically in reverse creation order, so resource cleanup belongs in each module's `destroy()`.

Because `define()` runs before `validate()`, an invalid block still displays its ports and editable fields so the user can fix the configuration. Any configuration-dependent `define()` must therefore read from `candidate()`, remain side-effect-free, and bound any candidate-controlled loops or port counts.

### Where to Place Validation

| Block Structure | Where Validation Belongs |
|---|---|
| One block wrapping one module | Put config and input validation in the module. The block defines ports, translates config, builds the module, and exposes outputs. |
| One block orchestrating multiple modules | Let each module validate its own config and inputs. Validate in the block only for interface choices, module selection, topology, or cross-module constraints. |
| A rule visible at both layers | Implement it once in the module and let module creation or validate-only reconfiguration surface the error through the block. |

Never repeat identical range, dtype, shape, or axis checks in a 1-to-1 block and module pair. Duplicate checks drift over time and produce inconsistent error messages depending on which layer runs first. The module is always the authoritative boundary because it can also be instantiated directly, used by another block, or implemented by multiple providers.

## Block States

| State | Meaning |
|---|---|
| **Created** | Everything succeeded and the block participates in compute cycles. |
| **Incomplete** | The block is valid yet waiting to run, either because a declared input is unconnected or unresolved, or because `create()` intentionally returned `Result::INCOMPLETE`. This is a normal waiting state, not an error. |
| **Errored** | A lifecycle hook failed, an undeclared input arrived, a declared output was missing, or a child module failed at runtime. The latest error message is kept as the block diagnostic. |

Validation failures keep the candidate configuration and declared interface intact so the node stays editable. When a child module fails during compute, its owning block transitions to **Errored** and downstream blocks recreate into **Incomplete**.

Returning `Result::INCOMPLETE` intentionally is the standard gating pattern for blocks waiting on late-arriving data, such as a server-supplied environment key. The flowgraph automatically destroys and recreates incomplete blocks whenever a new environment key becomes visible (see [Flowgraph Environment](/docs/metadata#flowgraph-environment)).

## Reconfiguration Sequence

```text
Block::reconfigure
├── Deserialize candidate configuration
├── define() candidate interface
├── validate() candidate semantics
├── Commit candidate configuration
├── configure() derived block state
├── Validate the edit in every child module
├── Apply the edit in every child module
├── reconfigure() the block (only if currently Created)
└── Publish the updated interface
```

How failures and rebuilds behave during reconfiguration:

- **Semantic validation error:** Accepted as a graph edit. The flowgraph rebuilds affected blocks with the candidate config, publishes the edited block as **Errored**, and keeps downstream wires as unresolved links so fixing the typo restores the graph cleanly.
- **Apply failure after validation:** The flowgraph restores the previous working graph state and returns the error.
- **In-place edit not possible:** Return `Result::RECREATE` (for instance, when a buffer size changes an allocation) so the flowgraph rebuilds the affected subgraph.

The block's own `reconfigure()` hook runs last, and only when the block is **Created** and every child module has already applied the edit. Use it for follow-up work that depends on updated module state (the candidate interface is still open for adjustments until `reconfigure()` returns). Prepare child module settings earlier in `configure()`, and detect structural rebuilds up front in `validate()` before modules apply anything. Incomplete or invalid blocks never enter `reconfigure()`, and `create()` never calls it either. The default implementation returns `Result::SUCCESS`.

## Block Macros and Helpers

Every `Block::Config` struct declares its metadata with five registration macros:

| Macro | Purpose |
|---|---|
| `JST_BLOCK_TYPE` | Sets the stable type string stored in YAML flowgraphs and used for registry lookups. |
| `JST_BLOCK_DOMAIN` | Assigns the block picker category (`Core`, `DSP`, `IO`, `Visualization`, or `ML`). |
| `JST_BLOCK_PARAMS` | Generates `serialize`, `deserialize`, and `hash` for the listed fields. Include every field affecting behavior so change detection stays accurate. Field types follow [Type Conversion](/docs/metadata#type-conversion). |
| `JST_BLOCK_NODE_SIZE` | Sets the default editor footprint (`XS`, `S`, `M`, `L`, or `XL`, defaulting to `S`). |
| `JST_BLOCK_DESCRIPTION` | Defines the node title, one-line picker summary, and full documentation pane text. |

Inside `Block::Impl`, use these helpers to build modules and declare the node interface:

| Helper | Purpose |
|---|---|
| `moduleCreate(name, config, inputs)` | Instantiates a child module with the block's device, runtime, and provider. |
| `moduleExposeOutput(blockPort, {module, modulePort})` | Connects a child module output to a declared block output (every output declared in `define()` must be exposed). |
| `moduleGetOutput({module, modulePort})` | Retrieves a child output to wire internal module chains. |
| `moduleBindConfigEdit(module, moduleKey, blockKey)` | Allows an interactive child module to propose edits to a block config field (see [Module-Originated Configuration Edits](/docs/module-contract#module-originated-configuration-edits)). |
| `defineInterfaceConfig()` / `defineInterfaceMetric()` | Declares editable controls and live readouts (see [Block Metrics](/docs/metadata#block-metrics)). |
| `updateInterfaceConfigFormat(key, format)` | Refines a control's format once runtime state exists (for example, dropdown options reported by a hardware module). Declare a placeholder format in `define()`, then update it from a shared helper called in `create()` and `reconfigure()`. |
| `environment()`, `view()`, `scheduler()`, `render()` | Accesses the flowgraph environment, metrics view, scheduler, and render window. |

## Interface Descriptions

Both `defineInterfaceConfig()` and `defineInterfaceMetric()` accept a `Parser::Map` as their fourth argument describing how the UI renders the field. Every descriptor requires a `type` string plus any type-specific properties. String literals inside map initializers convert to owned `std::string` values automatically. For sequences of string literals, use `Parser::MakeSequence({"bin", "raw", "iq"})` to produce owned strings while preserving other element types.

```cpp
JST_CHECK(defineInterfaceConfig("frequency", "Frequency", "Tuner frequency.", {
    {"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f},
    {"precision", 3}, {"step_config", "frequencyStep"},
}));

JST_CHECK(defineInterfaceConfig("mode", "Mode", "Select a processing mode.", {
    {"type", "dropdown"},
    {"options", Parser::Sequence{
        Parser::Map{{"label", "Fast"}, {"value", "fast"}},
        Parser::Map{{"label", "Accurate"}, {"value", "accurate"}},
    }},
}));
```

### Supported Control Types

| Control Type | Properties and Defaults |
|---|---|
| `bool`, `text`, `markdown`, `python`, `python-console` | No extra properties required. |
| `int`, `uint` | Optional `unit` suffix (defaults to `""`). |
| `float` | Defaults: `unit` `""`, `scale` `1`, `precision` `2`. Optional `step_config` links a step-size config key. Displayed values and steps divide by `scale`, while edits multiply by `scale`. Units are display labels only. |
| `range` | Defaults: `min` `0`, `max` `1`, `unit` `""`, `value_type` `float`. Setting `value_type` to `uint` changes default `max` to `100`. Bounds govern the slider only, while block and module validation enforces hard limits. |
| `dropdown` | Required `options` sequence of maps with string `label` and string `value`. Preserves order, supports duplicate labels via index, never splits on delimiters, displays unknown saved values as unavailable until changed, and converts string values during config deserialization. |
| `vector`, `vector-inline` | Defaults: `value_type` `float`, `unit` `""`, `scale` `1`, `precision` `2`. Also supports `uint` elements (and `value_type: "string"` on `vector`, which requires an `options` sequence of label/value maps). On `vector`, optional non-empty `item_label` customizes each element label. Scaling and precision apply to float elements. |
| `filepicker`, `filesave` | Optional `extensions` string sequence (defaults to empty). |
| `multiline` | Optional `collapsible` boolean (defaults to `false`). |
| `tensor-config` | Required `source` naming the config key with tensor specs. Optional `index` (defaults to `0`). |

Any config control except editor fields (`markdown`, `python`, `python-console`) also accepts an optional boolean `advanced` flag to group less common parameters under the block's **Advanced** section.

### Validation, Decoding, and Serialization Rules

- **Validated at registration.** Unknown types, malformed options, non-positive or non-finite scales, invalid bounds, precision outside `0..16`, unsupported payload types, and non-reflexive values (such as `NaN`) all return `Result::ERROR`.
- **Cached decoding and error notifications.** Controls cache decoded settings whenever descriptors change and decode values with `Parser::Deserialize`. Entering a decode-error state emits one notification without spamming repeated alerts, resetting once decoding succeeds again.
- **Typed accessors.** Calling `Parser::Get` reads validated properties via `Deserialize` and returns a fallback when missing or invalid, while central validation uses `Deserialize` directly to report malformed properties. Checked numeric conversions live in the Parser decoder, excluding booleans from numeric cross-conversion (including inside optionals and sequences).
- **Shared rendering and YAML boundaries.** Both inline node controls and detached surface panes share the same field construction path without encoding or decoding serialized documents during rendering. Use `Parser::YamlEncode` and `Parser::YamlDecode` strictly at serialization boundaries, where decoded scalars start as strings and convert during property deserialization.
