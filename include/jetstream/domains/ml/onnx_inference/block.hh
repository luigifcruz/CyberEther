#ifndef JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_BLOCK_HH
#define JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct OnnxInference : public Block::Config {
    std::string              modelPath         = "";
    std::vector<std::string> inputNames        = {"modelInput"};
    std::vector<std::string> outputNames       = {"output"};
    U64                      batchSize         = 1;
    std::string              executionProvider = "cpu";

    JST_BLOCK_TYPE(onnx_inference);
    JST_BLOCK_DOMAIN("ML");
    JST_BLOCK_PARAMS(modelPath, inputNames, outputNames, batchSize, executionProvider);
    JST_BLOCK_DESCRIPTION(
        "ONNX Inference",
        "Runs an ONNX model on input tensors.",
        "# ONNX Inference\n"
        "Loads an ONNX model and runs inference on every incoming tensor. "
        "Supports CPU, Core ML, and TensorRT execution providers.\n\n"

        "## Arguments\n"
        "- **Model Path**: Filesystem path to the `.onnx` model file.\n"
        "- **Input Names**: ONNX model input tensor name(s). One port is declared per entry.\n"
        "- **Output Names**: ONNX model output tensor name(s). One port is declared per entry.\n"
        "- **Batch Size**: Expected batch dimension — used when the model declares a dynamic batch dim.\n"
        "- **Execution Provider**: Hardware backend for inference. "
        "Core ML routes ops to the Apple Neural Engine on Apple Silicon. "
        "Switching to Core ML triggers a one-time model compile (~minutes for large models).\n\n"

        "## Inputs\n"
        "- **input** (single) or **input_0, input_1, …** (multiple): "
        "F32 tensor(s) matching the model's expected input shape(s).\n\n"

        "## Outputs\n"
        "- **output** (single) or **output_0, output_1, …** (multiple): "
        "F32 tensor(s) with the model's output shape(s)."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_BLOCK_HH
