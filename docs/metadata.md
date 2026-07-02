---
title: Metadata
description: Sharing values between blocks with the environment, tensor attributes, and metrics.
order: 81
category: Development
---

Tensors move the bulk data through a flowgraph, but most blocks also need to share small typed values: the sample rate attached to a stream, a calibration table delivered by a server, or a progress figure shown in the UI. CyberEther has three metadata channels for this, each with a different scope and direction. This page covers how to produce and consume each one from a native block, with pointers to the [Python Block](/docs/python-block) guide for the `compute(ctx)` view of the same data.

## Choosing a Channel

| | Environment | Tensor Attributes | Metrics |
|---|---|---|---|
| Scope | Whole flowgraph | One tensor, follows its connections | One block |
| Direction | Any block or host code, read and write | Producer writes, downstream reads | Owner publishes, anyone reads |
| Lifetime | Persists with the flowgraph | Lives with the tensor | Computed on demand |
| Typical use | Shared state, server-provided values, calibration | Signal description such as `sampleRate` and `frequency` | Progress, throughput, diagnostics |
| UI surface | Environment window | Pin tooltips | Node body |

A useful rule of thumb follows the data. If the value describes a specific stream, attach it to the tensor as an attribute so it travels with the connection. If it describes the state of one block and is only interesting to observers, publish it as a metric. If it is genuinely global, or must exist before blocks can start, put it in the environment.

All three channels carry the same value types. See [Type Conversion](/docs/python-block#type-conversion) for the full table: scalars up to 64 bits including complex, strings, typed vectors, and nested maps and sequences. None of them are meant for bulk data. A filter kernel of a few thousand taps is fine, a sample stream is not.

## Flowgraph Environment

The environment is a graph-wide, timestamped key-value store owned by the flowgraph. Every top-level key holds a map. Blocks reach it through `environment()` (available in `Block::Impl` and through the module context), host code through `flowgraph->environment()`, and Python code through `ctx.env`.

Typed access works through any `JST_SERDES` struct, which serializes to a nested map:

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

The `INCOMPLETE` return in this example is the intended pattern for blocks that depend on values delivered after the flowgraph opens, for example by a block that maintains a connection to a central server. The block sits in the incomplete state, and the flowgraph automatically destroys and recreates it when a new environment key becomes visible. Retries are triggered by key visibility changes only. In-place updates to an existing key do not recreate anything, so blocks that publish status values every cycle stay cheap.

Writing works symmetrically, and raw `Parser::Map` access is available when a struct is overkill:

```cpp
Parser::Map status;
status["locked"] = true;
status["snr"] = F32{23.4f};
JST_CHECK(environment()->set("receiver_status", status));
```

Points worth knowing:

- **Values are timestamped.** The full signature is `set(key, value, start, end)`, and `get(key, value, timestamp)` returns the newest entry whose range covers the timestamp. The defaults cover all time, which is the right choice unless you are recording values that change over a run.
- **The epoch tracks changes.** Every write increments `environment()->epoch()`. Poll it to refresh cached reads cheaply instead of deserializing keys every cycle. The Python bridge uses the same mechanism internally.
- **Keys persist and are visible.** Environment values are saved with the flowgraph and shown in the Environment window, so treat key names as a small public schema rather than scratch space.
- **Writers define the schema.** Python writes coerce to the type already stored, so a key seeded with exact widths, for example `F32` gain or a `vector<CF32>` of taps, keeps those widths across Python edits. Native `set()` calls replace the stored value verbatim, so C++ writers are responsible for publishing consistent types themselves.

The Python view of the environment, including the write tracking and rollback semantics, is covered in [Flowgraph Environment](/docs/python-block#flowgraph-environment).

## Tensor Attributes

Attributes are named values attached to a tensor, and they travel wherever the tensor is connected. Producers set them when the tensor is created or when the described property changes, and every downstream consumer reads them from its input:

```cpp
Result create() override {
    JST_CHECK(buffer.create(device(), DataType::CF32, {size}));
    buffer.setAttribute("sampleRate", sampleRate);
    buffer.setAttribute("frequency", frequency);
    outputs()["signal"].produced(name(), "signal", buffer);
    return Result::SUCCESS;
}
```

Consumers read with `attribute()` and friends:

```cpp
const auto& signal = inputs().at("signal").tensor;
if (signal.hasAttribute("sampleRate")) {
    const auto rate = std::any_cast<F64>(signal.attribute("sampleRate"));
}
```

Attributes do not propagate through a block automatically. A block that transforms a stream and wants to preserve its description copies what applies, either selectively with `setAttribute` or wholesale with `propagateAttributes(source)`, and then overrides what it changed. A decimator, for example, propagates the input attributes and rewrites `sampleRate`.

Hints:

- **Stick to the established names.** Blocks across the tree already use `sampleRate` and `frequency`, and the UI pin tooltips display whatever is attached. Inventing a second spelling for an existing concept breaks interoperability silently.
- **Attributes are per tensor, not per cycle.** They describe the stream, not the current buffer. Update them when the property changes, such as after a retune, and leave them alone otherwise. Change detection downstream can then key off the value cheaply.
- **Mind the stored width when reading.** An attribute holds exactly the type the producer stored, and `std::any_cast` throws on a mismatch, so an `F32` sample rate cannot be read as `F64`. Check the type or agree on widths for shared names. Blocks in the tree store `sampleRate` and `frequency` at different widths today, so defensive reads are the safe habit.

Python code sees the same values through `ctx.input_attrs` and `ctx.output_attrs`, described in [Tensor Attributes](/docs/python-block#tensor-attributes).

## Block Metrics

Metrics are read-only values a block publishes about itself. They are defined in `define()` alongside the block's inputs and outputs, and each metric is a callable evaluated when somebody looks:

```cpp
Result define() override {
    JST_CHECK(defineInterfaceMetric("progress",
                                    "Position",
                                    "Current file position.",
                                    "progressbar",
        [this]() -> std::any {
            const F32 progress = currentProgress();
            return std::pair<std::string, F32>{jst::fmt::format("{:.1f}%", progress * 100.0f), progress};
        }));

    JST_CHECK(defineInterfaceMetric("throughput",
                                    "Throughput",
                                    "Smoothed recent transfer rate.",
                                    "label",
        [this]() -> std::any {
            return jst::fmt::format("{:.1f} MB/s", currentBandwidth());
        }));

    return Result::SUCCESS;
}
```

The format string controls presentation and visibility:

- A format of `label` renders a `std::string` in the node body. Any other value type shows as an invalid metric in the UI, so format numbers into text for display.
- A format of `progressbar` expects a `std::pair<std::string, F32>` holding the display label and a fraction between 0 and 1.
- Formats prefixed with `private-` are internal by convention. The UI and the Python metrics mapping skip them, which makes them the right place for diagnostics. The filtering lives in those consumers, so C++ code reading through the view receives every metric and applies the convention itself.

Consumers read metrics through `Flowgraph::View::metrics(blockName, entries)` or, from Python, the subscription-based `ctx.metrics` mapping described in [Block Metrics](/docs/python-block#block-metrics).

Hints:

- **Keep the lambdas cheap and safe.** They run outside your compute path, potentially every UI frame, and on a different thread than compute. Return cached values updated by compute rather than computing on demand, and guard any shared state accordingly.
- **Displayed metrics are strings, programmatic values may deserve another channel.** A formatted string such as `"12.3 MB/s"` reads well in the node but forces every programmatic consumer to parse it, while a raw numeric metric reaches Python with its native type but renders as invalid in the label view. When a value matters to both audiences, publish the display string as the metric and mirror the raw number into the environment.
- **Metrics are pull, not push.** Nothing is stored or timestamped, and a metric that nobody reads costs nothing. If a value needs history or must outlive the block, it belongs in the environment instead.
