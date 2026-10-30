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
        "\n"
        "Runs a trained ONNX model once per buffer, with one port for each model input "
        "and output. It usually follows the blocks that shape a signal into what the "
        "model expects, such as Amplitude and Expand Dims.\n"
        "\n"
        "- **Types must match exactly.** The block never converts an input to the type "
        "the model declares.\n"
        "- **Output shapes are fixed at creation.** A dynamic output size comes from "
        "one trial run on zeros.\n"
        "- **Outputs start with no attributes.** Neither `sampleRate` nor the axis "
        "roles reach them.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Model input name** (in) | The model's type, see Setup | The model's rank, "
        "matching every fixed size. The buffer must be contiguous. One port per model "
        "input, named after it. |\n"
        "| **Model output name** (out) | The model's type | The model's shape, in a new"
        " buffer on the CPU. One port per model output, named after it. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Model Path** | Empty | An .onnx file | Model to run. Its tensors set the "
        "ports. |\n"
        "| **Execution Provider** | CPU | CPU, Core ML, TensorRT | Backend that runs "
        "the model. Core ML and TensorRT need a matching ONNX Runtime, see Setup. |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block shows no ports | Model Path is empty, the file cannot be read, or "
        "the model has no inputs | Pick a readable .onnx file with at least one input. "
        "|\n"
        "| The log warns that a model tensor uses an unsupported dtype, and the block "
        "stays incomplete | A model input or output uses a type outside Setup, such as "
        "F16 or a string | Export the model with supported tensor types. |\n"
        "| The block reports that a tensor has a dtype the model does not expect, or an"
        " unsupported dtype | The connected type differs from the model's, such as "
        "complex samples into an F32 input | Feed the declared type, such as F32 levels"
        " from an Amplitude block. |\n"
        "| The block reports an invalid rank or dimension for an input | The input "
        "shape misses a fixed size of the model, such as 1,024 samples into a model "
        "that expects 1 x 1,024 | Add the missing axis with Expand Dims, or regroup "
        "with Reshape. |\n"
        "| The block reports that the execution provider requires a provider that is "
        "not available | The ONNX Runtime build lacks Core ML, or lacks TensorRT or "
        "CUDA for TensorRT | Pick CPU, or use an ONNX Runtime build with that provider."
        " |\n"
        "\n"
        "## Setup\n"
        "\n"
        "| Platform | Requirement |\n"
        "|---|---|\n"
        "| **CPU** | A CyberEther build with ONNX Runtime. Without it, the block is not"
        " offered. |\n"
        "| **Core ML** | An ONNX Runtime build that includes the Core ML provider. |\n"
        "| **TensorRT** | An ONNX Runtime build that includes both the TensorRT and "
        "CUDA providers. |\n"
        "| Model file | At least one input, and tensors of F32, F64, or an integer type"
        " from I8 to U64. Every output size is known once the input sizes are. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block hands its buffers to ONNX Runtime, which runs the whole model each "
        "cycle. Core ML and TensorRT work inside ONNX Runtime, so every port stays on "
        "the CPU.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Python** runs custom NumPy code instead of a model.\n"
        "- **Cast** turns integer samples into F32.\n"
        "- **Expand Dims** adds the batch axis many models expect.\n"
        "- **Reshape** regroups an input into the model's shape."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_BLOCK_HH
