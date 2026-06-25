#ifndef JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_BLOCK_HH
#define JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct OnnxInference : public Block::Config {
    std::string modelPath = "";
    std::string executionProvider = "cpu";

    JST_BLOCK_TYPE(onnx_inference);
    JST_BLOCK_DOMAIN("ML");
    JST_BLOCK_PARAMS(modelPath, executionProvider);
    JST_BLOCK_DESCRIPTION(
        "ONNX Inference",
        "Runs an ONNX model on input tensors.",
        "# ONNX Inference\n"
        "Loads an ONNX model and runs inference on every incoming tensor. "
        "Supports CPU, Core ML, and TensorRT execution providers. "
        "Supported tensor dtypes include F32, F64, I8, I16, I32, I64, U8, U16, U32, and U64, and the connected tensors must match the model exactly. "
        "If a model reports a newer ONNX dtype that CyberEther cannot execute yet, the block keeps its ports visible but stays incomplete until you pick a supported model.\n\n"

        "## Arguments\n"
        "- **Model Path**: Filesystem path to the `.onnx` model file.\n"
        "- **Execution Provider**: Hardware backend for inference.\n\n"

        "## Useful For\n"
        "- Running trained ML models inside a flowgraph.\n"
        "- Connecting DSP preprocessing pipelines to neural-network inference.\n"

        "## Examples\n"
        "- Single-input model:\n"
        "  Ports are inferred from the selected ONNX model's graph inputs and outputs.\n"
        "  Input: input_0 F32[1, 1024] -> Output: output_0 F32[1, N]\n"
        "- Multi-input model:\n"
        "  Input: input_0 F32[1, 1024], input_1 F32[1, 8] -> Output: output_0 F32[1, N]\n\n"

        "## Implementation\n"
        "Inputs -> ONNX Runtime Session -> Outputs\n"
        "1. Reads ONNX graph input and output names from the selected model.\n"
        "2. Creates one input port and one output port per model tensor.\n"
        "3. Validates input and output tensor dtypes against the model, then allocates output buffers.\n"
        "4. Runs inference each cycle using the selected execution provider."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_BLOCK_HH
