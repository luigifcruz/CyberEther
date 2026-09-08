#ifndef JETSTREAM_DOMAINS_DSP_PHASE_CORRECTION_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_PHASE_CORRECTION_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct PhaseCorrection : public Block::Config {
    F64 phaseIncrement = 0.0;

    JST_BLOCK_TYPE(phase_correction);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(phaseIncrement);
    JST_BLOCK_DESCRIPTION(
        "Phase Correction",
        "Applies a phase rotation to a complex signal.",
        "# Phase Correction\n"
        "The Phase Correction block rotates a CF32 signal by a phase that advances "
        "for each batch and persists across processing submissions. If the signal "
        "has no `batchAxis`, each submission is treated as one batch. Signal shape "
        "and axis metadata are preserved.\n\n"

        "## Arguments\n"
        "- **Phase Increment**: Phase advance per batch in radians.\n\n"

        "## Useful For\n"
        "- Maintaining phase continuity across independently processed buffers.\n"
        "- Correcting deterministic phase progression introduced by resampling.\n\n"

        "## Implementation\n"
        "1. Multiply each batch by `exp(i * phase)`.\n"
        "2. Advance the phase by the configured increment for the next batch.\n"
        "3. Preserve the accumulated phase across processing submissions."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_PHASE_CORRECTION_BLOCK_HH
