#ifndef JETSTREAM_DOMAINS_ML_INFER_BLOCK_HH
#define JETSTREAM_DOMAINS_ML_INFER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Infer : public Block::Config {
    std::string modelPath = "";
    std::string inputName = "modelInput";
    std::string outputName = "output";
    U64 batchSize = 1;
    std::string executionProvider = "cpu";

    JST_BLOCK_TYPE(infer);
    JST_BLOCK_DOMAIN("ML");
    JST_BLOCK_PARAMS(modelPath, inputName, outputName, batchSize, executionProvider);
    JST_BLOCK_DESCRIPTION(
        "Infer",
        "Runs an ONNX model on the input tensor.",
        "# Infer\n"
        "Loads an ONNX model and runs inference on every incoming tensor. "
        "Supports CPU, Core ML, and TensorRT execution providers.\n\n"

        "## Arguments\n"
        "- **Model Path**: Filesystem path to the `.onnx` model file.\n"
        "- **Model Input Name**: ONNX model input tensor name (e.g. `modelInput`).\n"
        "- **Model Output Name**: ONNX model output tensor name (e.g. `output`).\n"
        "- **Batch Size**: Expected batch dimension — used for shape display.\n"
        "- **Execution Provider**: Hardware backend for inference. "
        "Core ML routes ops to the Apple Neural Engine on Apple Silicon. "
        "Switching to Core ML triggers a one-time model compile (~minutes for large models).\n\n"

        "## Inputs\n"
        "- **Input**: F32 tensor matching the model's expected input shape.\n\n"

        "## Outputs\n"
        "- **Output**: F32 tensor with the model's output shape."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_ML_INFER_BLOCK_HH
