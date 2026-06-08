#include <jetstream/domains/ml/infer/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/ml/infer/module.hh>

namespace Jetstream::Blocks {

struct InferImpl : public Block::Impl, public DynamicConfig<Blocks::Infer> {
    Result configure() override;
    Result define() override;
    Result create() override;

  protected:
    std::shared_ptr<Modules::Infer> moduleConfig = std::make_shared<Modules::Infer>();
};

Result InferImpl::configure() {
    moduleConfig->modelPath         = modelPath;
    moduleConfig->inputName         = inputName;
    moduleConfig->outputName        = outputName;
    moduleConfig->batchSize         = batchSize;
    moduleConfig->executionProvider = executionProvider;
    return Result::SUCCESS;
}

Result InferImpl::define() {
    JST_CHECK(defineInterfaceInput("input",
                                   "Input",
                                   "F32 tensor matching the model's expected input shape."));
    JST_CHECK(defineInterfaceOutput("output",
                                    "Output",
                                    "F32 tensor with the model's output shape."));

    JST_CHECK(defineInterfaceConfig("modelPath",
                                    "Model Path",
                                    "Filesystem path to the .onnx model file.",
                                    "text"));
    JST_CHECK(defineInterfaceConfig("inputName",
                                    "Input Name",
                                    "ONNX graph input node name.",
                                    "text"));
    JST_CHECK(defineInterfaceConfig("outputName",
                                    "Output Name",
                                    "ONNX graph output node name.",
                                    "text"));
    JST_CHECK(defineInterfaceConfig("batchSize",
                                    "Batch Size",
                                    "Expected batch dimension (for display only).",
                                    "int:batches"));
    JST_CHECK(defineInterfaceConfig("executionProvider",
                                    "Execution Provider",
                                    "Hardware backend used for inference. "
                                    "CoreML routes ops to the Apple Neural Engine on Apple Silicon. "
                                    "First switch to CoreML triggers a one-time model compile (~minutes for large models).",
                                    "dropdown:CPU(cpu),CoreML / ANE(coreml),CUDA + TensorRT(cuda_trt)"));

    return Result::SUCCESS;
}

Result InferImpl::create() {
    JST_CHECK(moduleCreate("infer", moduleConfig, {
        {"input", inputs().at("input")},
    }));
    JST_CHECK(moduleExposeOutput("output", {"infer", "output"}));
    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(InferImpl);

}  // namespace Jetstream::Blocks
