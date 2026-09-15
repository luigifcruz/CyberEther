#ifndef JETSTREAM_DOMAINS_DSP_RATIONAL_RESAMPLER_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_RATIONAL_RESAMPLER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct RationalResampler : public Block::Config {
    U64 interpolation = 1;
    U64 decimation = 1;
    U64 taps = 0;
    F32 cutoff = 0.9f;

    JST_BLOCK_TYPE(rational_resampler);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(interpolation, decimation, taps, cutoff);
    JST_BLOCK_DESCRIPTION(
        "Rational Resampler",
        "Resamples a signal by a rational factor.",
        "# Rational Resampler\n"
        "The Rational Resampler changes the sample rate by Interpolation / "
        "Decimation. Both factors are positive integers and are reduced by their "
        "greatest common divisor before designing the filter. A Blackman-windowed "
        "low-pass filter suppresses aliases and interpolation images while "
        "preserving the signal's DC gain.\n\n"

        "The CPU implementation accepts F32 and CF32 signals with `sampleAxis` "
        "metadata (axis 0 is inferred for one-dimensional input). Channels and "
        "other independent lanes keep separate history. "
        "Batches are consecutive segments of each lane. Axis roles, dtype, and "
        "frequency metadata are preserved. The `sampleRate` value, when present as F32, "
        "is multiplied by the rate ratio. Sample-rate metadata is refreshed at "
        "creation and on each processing submission, including late publication "
        "and removal.\n\n"

        "## Arguments\n"
        "- **Interpolation**: Numerator of the output/input sample-rate ratio.\n"
        "- **Decimation**: Denominator of the output/input sample-rate ratio.\n"
        "- **Taps**: Odd FIR length at the interpolated rate. Zero selects "
        "64 * max(L, M) + 1 taps, where L and M are the reduced factors. "
        "An explicit length must be at least 2 * max(L, M) + 1.\n"
        "- **Cutoff**: Filter cutoff as a fraction of the smaller input/output "
        "Nyquist frequency, strictly between 0 and 1. Defaults to 0.9.\n\n"

        "## Streaming Behavior\n"
        "An N-sample input produces output tensors with ceil(N * L / M) samples "
        "per batch. Buffers need not divide evenly by M. The fractional phase "
        "and filter history persist across submissions. Bounded queues retain output "
        "until a complete tensor is available. Incomplete tensors skip downstream "
        "processing for that cycle. No padding samples are emitted to fill them.\n\n"
        "The filter starts with zero history and has a group delay of "
        "(taps - 1) / (2 * M) output samples. Even a 1:1 ratio applies the FIR. "
        "Configuration changes recreate the module and reset its streaming state. "
        "At end of input, a partial output tensor and the unflushed filter tail "
        "remain buffered.\n\n"

        "## Examples\n"
        "- Convert 250 kS/s to 48 kS/s with Interpolation=24 and Decimation=125.\n"
        "- Convert a real signal with Interpolation=3 and Decimation=2 "
        "from F32[1024] to F32[1536].\n"
        "- Convert a complex signal with Interpolation=2 and Decimation=3 "
        "from CF32[2, 1024] to CF32[2, 683] "
        "with channels on axis 0 and samples on axis 1."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_RATIONAL_RESAMPLER_BLOCK_HH
