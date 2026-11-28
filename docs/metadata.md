---
title: Metadata
description: Sharing values between blocks with the environment, tensor attributes, metrics, and C++/Python type conversion.
order: 340
category: Development
---

Tensors carry bulk sample streams through a flowgraph, while most real-world blocks also need to exchange lightweight typed values: a stream's sample rate, a calibration table fetched from a server, or a live throughput readout for the UI.

CyberEther provides three distinct metadata channels, each tailored to a different scope and direction. This guide covers how native C++ blocks produce and consume all three, how Note blocks interpolate live environment values, and how metadata converts between C++ and Python.

## Choosing a Channel

| Property | [Environment](#flowgraph-environment) | [Tensor Attributes](#tensor-attributes) | [Metrics](#block-metrics) |
|---|---|---|---|
| **Scope** | Whole flowgraph | One tensor, following its wires | One block |
| **Direction** | Any block or host code (read/write) | Producer writes, downstream reads | Owner publishes, anyone reads |
| **Lifetime** | Persists for the flowgraph's lifetime | Travels with the tensor | Computed lazily on demand |
| **Best for** | Shared state, server values, calibration | Stream properties (`sampleRate`, `frequency`) | Progress, throughput, diagnostics |
| **Where it shows in UI** | Environment tile | Wire and pin tooltips | Block node body |

Let the data guide your choice:

- If a value describes a specific signal stream, attach it as a **tensor attribute** so it follows the wire.
- If a value reports one block's internal status for observers, publish it as a **metric**.
- If a value is global across the graph or must arrive before blocks start, store it in the **environment**.

All three channels share the same typed value system (see [Type Conversion](#type-conversion)): scalars up to 64 bits including complex numbers, strings, typed vectors, and nested maps and sequences. None of these channels are meant for high-rate bulk data. A filter kernel of a few thousand taps is fine, while a live sample stream belongs in a tensor.

## Flowgraph Environment

The environment is a graph-wide, timestamped key-value store owned by the flowgraph where every top-level key holds a map. Access it via `environment()` inside `Block::Impl` or a module context, `flowgraph->environment()` from host code, and `ctx.env` from Python.

### Typed Reads and the `INCOMPLETE` Gating Pattern

Any struct annotated with `JST_SERDES` serializes directly to and from a nested map:

```cpp
struct StationInfo {
    F64 frequency = 0.0;
    F32 gain = 1.0f;
    std::string callsign;

    JST_SERDES(frequency, gain, callsign);
};

Result create() override {
    StationInfo station;
    if (!environment()->tryGet("station", station)) {
        JST_ERROR("[MY_BLOCK] Waiting for 'station' environment value.");
        return Result::INCOMPLETE;
    }

    JST_CHECK(moduleCreate("source", makeConfig(station), {}));
    return Result::SUCCESS;
}
```

Returning `Result::INCOMPLETE` here is the standard gating pattern for blocks that wait on values delivered after startup (such as configuration arriving from a server). The block rests cleanly in the incomplete state, and the flowgraph automatically destroys and recreates it the moment a new environment key becomes visible. Updating an existing key in place never triggers recreation, keeping per-cycle status updates cheap.

### Writing Environment Entries

Writing works symmetrically with typed structs, or with a raw `Parser::Map` when defining a struct is unnecessary:

```cpp
Parser::Map status;
status["locked"] = true;
status["snr"] = F32{23.4f};
JST_CHECK(environment()->set("receiver_status", status));
```

Key details to keep in mind:

- **Entries are timestamped.** The full API is `set(key, value, start, end)` and `get(key, value, timestamp)`, returning the newest entry covering that timestamp. The defaults span all time, which is right for most use cases unless you record time-varying history.
- **Poll `epoch()` for cheap change detection.** Every write increments `environment()->epoch()`, letting you refresh cached values only when something changed instead of deserializing every cycle. The Python bridge uses the same mechanism.
- **Treat keys as a public schema.** Environment values are visible in the **Environment** tile, so choose clear, stable names rather than using the environment as scratch space.
- **Keep types consistent.** Python writes coerce to whatever type is already stored, so seeding a key with exact C++ widths (such as `F32` gain or `std::vector<CF32>` taps) preserves those widths across Python edits. Native `set()` calls overwrite the stored value verbatim, so C++ writers are responsible for maintaining consistent types.

For Python write tracking and rollback details, see [Flowgraph Environment](/docs/python-context#flowgraph-environment) in the Python Context API guide.

### Live Markdown in Note Blocks

Note blocks can interpolate live environment values directly into formatted Markdown using `${env.<path>}` placeholders with optional Python-style format specifiers:

```markdown
The SNR is **${env.receiver_status.snr:.1f} dB** and the lock state is ${env.receiver_status.locked}.
```

Paths navigate nested fields with dots and sequence elements with brackets (including negative indices counting from the end). Unavailable values display as `--`.

Use a `[*]` wildcard to repeat a line once per element, turning a list into a live table:

```markdown
| Station | Freq (MHz) |
|---|---:|
| ${env.stations.list[*].name} | ${env.stations.list[*].freq:.3f} |
```

Use a fenced `stats` block to render dashboard tiles (one per line: `Label | Value | Optional Tone` such as `green` or `red`):

````markdown
```stats
SNR | ${env.receiver.snr:.1f} dB | green
Uptime | ${env.receiver.uptime:.0f} s
```
````

## Tensor Attributes

Attributes are named values attached to a tensor that travel along every wire connected to it. Producers set them during `create()` (or whenever a stream property changes), and downstream blocks read them from their input tensors:

```cpp
Result create() override {
    JST_CHECK(buffer.create(device(), DataType::CF32, {size}));
    JST_CHECK(SetSignalAxes(buffer, {.sample = Index{0}}));
    buffer.setAttribute("sampleRate", sampleRate);
    buffer.setAttribute("frequency", frequency);
    outputs()["signal"].produced(name(), "signal", buffer);
    return Result::SUCCESS;
}
```

Consumers inspect attributes with `hasAttribute()` and `attribute()`:

```cpp
const auto& signal = inputs().at("signal").tensor;
if (signal.hasAttribute("sampleRate")) {
    const auto rate = std::any_cast<F64>(signal.attribute("sampleRate"));
}
```

Attributes do not propagate across blocks automatically. A processing block that wants to preserve upstream metadata copies what still applies, either selectively with `setAttribute()` or wholesale with `propagateAttributes(source)`, and then updates anything it changed. A decimator, for instance, propagates input attributes and divides `sampleRate`.

### Sample Scale

The `sampleScale` attribute tells downstream consumers how to interpret integer samples as normalized floating-point values. It is a finite, positive `F32` multiplier applied after unsigned integer types subtract their midpoint (such as `128` for `CU8`).

A producer of raw integer samples sets `sampleScale` to `1 / fullScale` (for example, `1/2048` for 12-bit samples stored in `CI16`) and keeps it constant for the lifetime of the stream.

The **Cast** block honors `sampleScale` automatically:

- **Integer to float:** Multiplies samples by `sampleScale` and writes `sampleScale = 1` on the float output.
- **Integer to integer:** Adjusts `sampleScale` on the output to preserve the same normalized signal level in the new integer type.
- **Float to integer:** Sets `sampleScale` to one over the full scale of the destination integer type.
- **Missing attribute:** Falls back to the full scale of the storage type.

### Signal Axes

Signal tensors describe the meaning of their dimensions through three standard attributes:

- The `sampleAxis` attribute identifies the sample dimension processed by sequence-oriented DSP blocks. A rank-one tensor with no axis attributes implicitly treats axis `0` as its sample axis.
- The `batchAxis` attribute optionally identifies ordered frames processed together.
- The `channelAxis` attribute optionally identifies independent channels. A channel-only tensor without samples may set only `channelAxis` (plus an optional `batchAxis`), and consumers explicitly declare whether they accept channel-only layouts.

No other axis semantics are inferred from tensor shape. Every axis index must be stored as an exact zero-based `Index` (`U64`, never `I64`), within the tensor's rank, with no two roles pointing to the same dimension. For example, a `[batch, channels, samples]` tensor sets `batchAxis=Index{0}`, `channelAxis=Index{1}`, and `sampleAxis=Index{2}`. Rank-one producers may still publish `sampleAxis=Index{0}` explicitly.

Use the framework helpers instead of guessing from tensor rank:

- **Sample-oriented consumers** call `ResolveSignalAxes()` and process `sampleAxis` independently across other dimensions unless they explicitly sequence batches or combine channels.
- **Channel-only consumers** validate optional roles with `MapSignalAxes()`.
- **Producers and shape-preserving modules** publish or preserve roles with `SetSignalAxes()`.

#### The Signal Axes Block

Users can assign or fix axis roles in a flowgraph without copying or reordering data using the **Signal Axes** block with positional `[B, C, S, _, *]` notation:

- Letters `B`, `C`, and `S` assign batch, channel, and sample roles.
- An underscore (`_`) leaves a dimension unlabeled.
- An asterisk (`*`) preserves the valid input role already on that dimension.

Each role may appear at most once, omitted trailing dimensions carry no explicit role, and a layout containing only `_` clears all three roles. Conflicting assignments (such as assigning a role on one axis while inheriting the same role via `*` on another) are rejected. An untagged rank-one output still remains implicitly sampled. Leaving the layout empty validates and preserves all input metadata unchanged, whereas an explicit layout without `*` can replace malformed input roles. To change tensor shape or dimension order, pair it with structural blocks like Slice, Squeeze Dims, or Permutation.

Structural tensor blocks remain explicitly axis-configured because modifying dimensions is their job. Built-in structural blocks remap `sampleAxis`, `batchAxis`, and `channelAxis` to the output layout (or drop a role when its dimension disappears or becomes ambiguous). Custom structural blocks must do the same rather than copying stale axis indices.

### Frame Boxes

The `boxes` attribute lists regions for the **Frame** block to outline over the image. It is a sequence of maps, each with numeric `x`, `y`, `width`, and `height` in frame pixels (row 0 at the top), plus an optional `label` string and an optional `color` given as three or four numbers from 0 to 1 or as `"#RRGGBB[AA]"`. Frame draws up to 64 boxes, skips malformed entries with a warning, and clears the overlay when the attribute is an empty sequence or absent. Unlike stream properties, `boxes` describes the current frame, so producers update it with every frame they emit:

```python
ctx.output_attrs[0]["boxes"] = [
    {"x": 120, "y": 40, "width": 30, "height": 12, "label": "pulse"},
]
```

### Attribute Best Practices

- **Reuse established names.** Built-in blocks and UI pin tooltips expect standard names like `sampleRate` and `frequency`. Inventing alternate spellings silently breaks interoperability.
- **Update only on stream changes.** Attributes describe the stream, not individual frames. Update them when a property actually changes (such as retuning a radio) so downstream change detection stays fast. The Frame `boxes` attribute is the exception, since it describes one frame.
- **Guard against width mismatches.** Because `std::any_cast` throws if types differ, an `F32` attribute cannot be read as `F64`. Agree on widths for shared names and read defensively, as some blocks in the tree currently store `sampleRate` and `frequency` at different widths.
- **Remember concrete Python types.** Assigning `np.float32` to `ctx.output_attrs` stores `F32`, while plain Python `float` stores `F64`, replacing the previous type rather than coercing to it (see [Tensor Attributes](/docs/python-context#tensor-attributes)).

## Block Metrics

Metrics are read-only status values a block publishes about itself. Register them in `define()` alongside ports and config fields, providing a callback that computes the current value on demand:

```cpp
Result define() override {
    JST_CHECK(defineInterfaceMetric("progress",
                                    "Position",
                                    "Current file position.",
                                    {{"type", "progressbar"}},
        [this]() -> std::any {
            const F32 progress = currentProgress();
            return std::pair<std::string, F32>{jst::fmt::format("{:.1f}%", progress * 100.0f), progress};
        }));

    JST_CHECK(defineInterfaceMetric("throughput",
                                    "Throughput",
                                    "Smoothed recent transfer rate.",
                                    {{"type", "label"}},
        [this]() -> std::any {
            return jst::fmt::format("{:.1f} MB/s", currentBandwidth());
        }));

    return Result::SUCCESS;
}
```

### Supported Display Formats

| Format `type` | Expected Return Value | Options and Behavior |
|---|---|---|
| `label` | `std::string` | Displays formatted text such as `"12.3 MB/s"` or `"Connected"`. Convert numbers to strings first. |
| `progressbar` | `std::pair<std::string, F32>` | Displays a label and a fill fraction from `0` to `1`. Add `warning_above` or `error_above` (higher is worse), or `warning_below` or `error_below` (lower is worse) to color the bar amber or red. Return `NaN` when no reading is available yet to keep the bar empty and uncolored. |
| `table` | `Parser::Map` with `columns` and `rows` | Renders tabular readings with one cell per column per row. Build vectors with `Parser::Serialize`, and note that cell strings may include tabs and line breaks. |

Metrics render inside the block node by default. Set `visibility` to `internal` for metrics meant for dedicated views (such as runtime timing or Python diagnostics). Internal metrics stay hidden from the node body while remaining readable from C++ and from Python (when convertible). The `visibility` property applies to metrics only.

Other C++ code reads metrics via `Flowgraph::View::metrics(blockName, entries)`. Python scripts read them through `ctx.metrics` (see [Block Metrics](/docs/python-context#block-metrics)).

### Metric Best Practices

- **Keep callbacks fast and thread-safe.** Metric lambdas run outside the compute path on the UI thread, potentially every frame. Return cached values updated by compute and guard shared state appropriately.
- **Separate human display from machine data.** Formatted strings like `"12.3 MB/s"` look great in nodes though they force code consumers to parse text, whereas raw numbers reach Python cleanly though they fail in a `label` widget. When both audiences need a value, publish the formatted string as a metric and mirror the raw number into the environment.
- **Metrics are pull-only.** Nothing is stored or timestamped, and an unread metric costs nothing. If a value needs history or must outlive the block, store it in the environment instead.

## Type Conversion

Metadata crossing between C++ and Python (in `ctx.env`, `ctx.input_attrs`, `ctx.output_attrs`, and metric `"value"` fields) converts using the rules below. When NumPy is importable, numeric types keep their exact bit widths across round trips. Without NumPy, reads fall back to plain Python types.

### C++ to Python (Reading)

| C++ Type | Python (With NumPy) | Python (Fallback) |
|---|---|---|
| `F32`, `F64` | `np.float32`, `np.float64` | `float` |
| `I8` to `I64`, `U8` to `U64` | `np.int8` to `np.uint64` | `int` |
| `CF32`, `CF64` | `np.complex64`, `np.complex128` | `complex` |
| `bool` | `bool` | `bool` |
| `std::string` | `str` | `str` |
| `std::vector<U64/F32/F64/CF32/CF64>` | Read-only `np.ndarray` of matching dtype | `tuple` |
| `Parser::Map` | `dict` (recursive) | `dict` (recursive) |
| `Parser::Sequence` | `tuple` (recursive) | `tuple` (recursive) |
| null | `None` | `None` |

Typed vectors transfer as contiguous raw buffers in both directions (one buffer copy, zero per-element Python objects). Because returned arrays are read-only, copy them before modifying.

### Python to C++ (Writing)

- **Existing environment entries keep their schema.** Writing a number onto an existing environment entry coerces it to the stored width: an `F32` stays `F32`, integers are range-checked strictly, floats only fit integer slots when integral, and complex values with non-zero imaginary parts are rejected for real slots. Rejected writes roll back with a console warning rather than silently truncating. Writing a different category (such as a number over a string or a dict over a scalar) replaces the entry.
- **Output attributes adopt the assigned type.** Just like C++ `setAttribute`, assigning to `ctx.output_attrs` replaces both value and type (`np.float32` stores `F32`, plain `float` stores `F64`).
- **New environment entries use native widths.** Plain Python values map to `bool`, `I64` (`U64` above signed range), `F64`, `CF64`, and `str`. NumPy scalars keep their exact width (`np.float32` to `F32`, `np.int16` to `I16`, `np.complex64` to `CF32`). Homogeneous lists or 1D arrays of floats and complex numbers become typed vectors of the matching width. Non-negative integer collections become `std::vector<U64>` (no narrower integer vector exists), while mixed collections become generic sequences.
- **Writing `None` always stores null**, whether at the top level of a value, inside a dict, or inside a sequence.

Multi-dimensional arrays, half-precision floats (use tensors for bulk data), and framework types (`Range`, `Extent2D`, `Tensor`, and device enums) are not supported in metadata. They are skipped on read and rejected with a console warning on write.
