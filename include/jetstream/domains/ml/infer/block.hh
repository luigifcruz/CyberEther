#ifndef JETSTREAM_DOMAINS_ML_INFER_BLOCK_HH
#define JETSTREAM_DOMAINS_ML_INFER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Infer : public Block::Config {
    std::string modelPath;
    std::string inputName         = "modelInput";
    std::string outputName        = "output";
    U64         batchSize         = 1;
    std::string executionProvider = "cpu";

    JST_BLOCK_TYPE(infer);
    JST_BLOCK_DOMAIN("ML");
    JST_BLOCK_PARAMS(modelPath, inputName, outputName, batchSize, executionProvider);
    JST_BLOCK_DESCRIPTION(
        "Infer",
        "Runs an ONNX model on the input tensor.",
        "# Infer\n"
        "Loads an ONNX model and runs inference on every incoming tensor. "
        "Supports CPU (universal), CoreML/ANE (Apple Silicon), and "
        "TensorRT+CUDA (NVIDIA) execution providers.\n\n"

        "## Arguments\n"
        "- **Model Path**: Filesystem path to the `.onnx` model file.\n"
        "- **Input Name**: ONNX graph input node name (e.g. `modelInput`).\n"
        "- **Output Name**: ONNX graph output node name (e.g. `output`).\n"
        "- **Batch Size**: Expected batch dimension — used for shape display.\n"
        "- **Execution Provider**: Hardware backend for inference. "
        "CoreML routes ops to the Apple Neural Engine on Apple Silicon. "
        "Switching to CoreML triggers a one-time model compile (~minutes for large models).\n\n"

        "## Inputs\n"
        "- **Input**: F32 tensor matching the model's expected input shape.\n\n"

        "## Outputs\n"
        "- **Output**: F32 tensor with the model's output shape."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_ML_INFER_BLOCK_HH
