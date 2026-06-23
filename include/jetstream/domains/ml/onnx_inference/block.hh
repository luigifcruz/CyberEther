#ifndef JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_BLOCK_HH
#define JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct OnnxInference : public Block::Config {
    std::string modelPath = "";
    U64 modelInputCount = 1;
    U64 modelOutputCount = 1;
    std::vector<std::string> inputNames = {"modelInput"};
    std::vector<std::string> outputNames = {"output"};
    U64 batchSize = 1;
    std::string executionProvider = "cpu";

    JST_BLOCK_TYPE(onnx_inference);
    JST_BLOCK_DOMAIN("ML");
    JST_BLOCK_PARAMS(modelPath, modelInputCount, modelOutputCount,
                     inputNames, outputNames, batchSize, executionProvider);
    JST_BLOCK_DESCRIPTION(
        "ONNX Inference",
        "Runs an ONNX model on input tensors.",
        "# ONNX Inference\n"
        "Loads an ONNX model and runs inference on every incoming tensor. "
        "Supports CPU, Core ML, and TensorRT execution providers.\n\n"

        "## Arguments\n"
        "- **Model Path**: Filesystem path to the `.onnx` model file.\n"
        "- **Model Input Count**: Number of ONNX model input tensors.\n"
        "- **Model Output Count**: Number of ONNX model output tensors.\n"
        "- **Model Input Tensor**: ONNX model input tensor name(s). One port is declared per entry.\n"
        "- **Model Output Tensor**: ONNX model output tensor name(s). One port is declared per entry.\n"
        "- **Batch Size**: Expected batch dimension for models with batched inputs.\n"
        "- **Execution Provider**: Hardware backend for inference.\n\n"

        "## Useful For\n"
        "- Running trained ML models inside a flowgraph.\n"
        "- Connecting DSP preprocessing pipelines to neural-network inference.\n"

        "## Examples\n"
        "- Single-input model:\n"
        "  Config: Model Input Count=1, Model Output Count=1, Model Input Tensor=input, "
        "Model Output Tensor=output\n"
        "  Input: input_0 F32[1, 1024] -> Output: output_0 F32[1, N]\n"
        "- Multi-input model:\n"
        "  Config: Model Input Count=2, Model Output Count=1, Model Input Tensor=[samples, metadata]\n"
        "  Input: input_0 F32[1, 1024], input_1 F32[1, 8] -> Output: output_0 F32[1, N]\n\n"

        "## Implementation\n"
        "Inputs -> ONNX Runtime Session -> Outputs\n"
        "1. Creates one input port and one output port per configured tensor count.\n"
        "2. Loads the ONNX model and resolves configured tensor names against the session.\n"
        "3. Validates F32 input and output tensors, then allocates output buffers.\n"
        "4. Runs inference each cycle using the selected execution provider."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_BLOCK_HH
