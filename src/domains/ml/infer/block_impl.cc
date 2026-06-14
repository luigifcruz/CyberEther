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
    moduleConfig->modelPath = modelPath;
    moduleConfig->inputName = inputName;
    moduleConfig->outputName = outputName;
    moduleConfig->batchSize = batchSize;
    moduleConfig->executionProvider = executionProvider;

    return Result::SUCCESS;
}

Result InferImpl::define() {
    JST_CHECK(defineInterfaceInput("input",
                                   "Input",
                                   "Tensor matching the model's expected input shape."));
    JST_CHECK(defineInterfaceOutput("output",
                                    "Output",
                                    "Tensor with the model's output shape."));

    JST_CHECK(defineInterfaceConfig("modelPath",
                                    "Model Path",
                                    "Filesystem path to the .onnx model file.",
                                    "filepicker:onnx"));
    JST_CHECK(defineInterfaceConfig("inputName",
                                    "Model Input Name",
                                    "ONNX model input tensor name.",
                                    "text"));
    JST_CHECK(defineInterfaceConfig("outputName",
                                    "Model Output Name",
                                    "ONNX model output tensor name.",
                                    "text"));
    JST_CHECK(defineInterfaceConfig("batchSize",
                                    "Batch Size",
                                    "Expected batch dimension.",
                                    "int:batches"));
    JST_CHECK(defineInterfaceConfig("executionProvider",
                                    "Execution Provider",
                                    "Execution backend for running the ONNX model.",
                                    "dropdown:cpu(CPU),coreml(Core ML),tensorrt(TensorRT)"));

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
