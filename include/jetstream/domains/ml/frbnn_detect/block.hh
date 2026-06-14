#ifndef JETSTREAM_DOMAINS_ML_FRBNN_DETECT_BLOCK_HH
#define JETSTREAM_DOMAINS_ML_FRBNN_DETECT_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct FrbnnDetect : public Block::Config {
    F32 threshold  = 0.5f;
    U64 classIndex = 0;

    JST_BLOCK_TYPE(frbnn_detect);
    JST_BLOCK_DOMAIN("ML");
    JST_BLOCK_PARAMS(threshold, classIndex);
    JST_BLOCK_DESCRIPTION(
        "FRBNN Detect",
        "Detects Fast Radio Burst candidates from FRBNN inference output.",
        "# FRBNN Detect\n"
        "Thresholds the probability tensor produced by the Infer block running "
        "an FRBNN model and emits a probability stream suitable for visualization. "
        "Logs candidate detections whenever the FRB class probability exceeds the "
        "configured threshold.\n\n"

        "## Arguments\n"
        "- **Threshold**: Probability above which a sample is flagged as an FRB candidate.\n"
        "- **Class Index**: Which output column of the model represents the FRB class "
        "(0-based; ignored when the input is 1-D).\n\n"

        "## Inputs\n"
        "- **Probabilities**: F32 tensor `[batch]` or `[batch, n_classes]` from the "
        "Infer block.\n\n"

        "## Outputs\n"
        "- **Signal**: F32 tensor `[batch]` — FRB-class probability per sample, ready "
        "for a LineplotBlock.\n\n"

        "## Notes\n"
        "Accepts both 1-D and 2-D probability tensors. In the 1-D case the single "
        "value per sample is used directly; in the 2-D case the column at `classIndex` "
        "is extracted."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_ML_FRBNN_DETECT_BLOCK_HH
