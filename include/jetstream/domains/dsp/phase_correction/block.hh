#ifndef JETSTREAM_DOMAINS_DSP_PHASE_CORRECTION_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_PHASE_CORRECTION_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct PhaseCorrection : public Block::Config {
    F64 phaseIncrement = 0.0;

    JST_BLOCK_TYPE(phase_correction);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(phaseIncrement);
    JST_BLOCK_DESCRIPTION(
        "Phase Correction",
        "Applies a phase rotation to a complex signal.",
        "# Phase Correction\n"
        "\n"
        "Rotates each batch of a complex signal by a phase that grows by a fixed step "
        "from one batch to the next. It usually follows a block that leaves a known "
        "phase step between consecutive batches.\n"
        "\n"
        "- **The phase carries over between buffers.** Batches across buffers form one "
        "sequence, starting from no rotation.\n"
        "- **Unbatched input rotates as one batch.** Without a `batchAxis`, each buffer"
        " gets a single rotation.\n"
        "- **Every head gets the same rotation.** Heads on a `channelAxis` share one "
        "phase.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | CF32 | Needs a `sampleAxis`, implied for one-dimensional "
        "input. |\n"
        "| **Output** (out) | CF32 | Same shape, with every axis and attribute kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Phase Increment** | 0 rad | Any | Rotation added for each batch, "
        "counterclockwise for positive values. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Pass through | Defaults | 8 batches x 1,024 samples | The same values |\n"
        "| Quarter turn per batch | Phase Increment 1.571 rad | 2 batches x 3 samples, "
        "all 1 | First buffer: 1, then j. Second buffer: -1, then -j. |\n"
        "| Slow drift per buffer | Phase Increment 0.1 rad | 8,192 samples | Buffer n "
        "rotated by 0.1 n rad |\n"
        "| Undo a step of 0.5 rad | Phase Increment -0.5 rad | 3 heads x 8 batches x "
        "1,024 samples | Batch k of every head rotated back by 0.5 k rad |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The phase error doubles instead of vanishing | Phase Increment has the same "
        "sign as the step to undo | Negate Phase Increment. |\n"
        "| The whole buffer rotates as one | The input has no `batchAxis` | Split it "
        "into batches with Reshape, then assign roles with Signal Axes. |\n"
        "| The block reports that the input must be CF32 | The input is real or another"
        " type | Convert it with a Cast block. |\n"
        "| The block reports that the input signal axis metadata is invalid | The input"
        " has more than one dimension and no `sampleAxis` | Assign roles with a Signal "
        "Axes block. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each batch is multiplied by a unit phasor whose angle grows by the increment "
        "from batch to batch. The angle is kept wrapped and carries on into the next "
        "buffer. The model below covers one buffer of batches by samples, with the "
        "phase passed in and returned.\n"
        "\n"
        "```python\n"
        "import math\n"
        "import numpy as np\n"
        "\n"
        "def phase_correction(x, phase=0.0, phase_increment=0.0):\n"
        "    step = math.remainder(phase_increment, 2 * math.pi)\n"
        "    angles = phase + step * np.arange(x.shape[0])\n"
        "    y = x * np.exp(1j * angles)[:, None]\n"
        "    phase = math.remainder(phase + step * x.shape[0], 2 * math.pi)\n"
        "    return y.astype(np.complex64), phase\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Filter Engine** applies this correction itself when resampling.\n"
        "- **Signal Axes** adds the `batchAxis` it steps over.\n"
        "- **Cast** converts real input to CF32."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_PHASE_CORRECTION_BLOCK_HH
