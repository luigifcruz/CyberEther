#include <jetstream/domains/dsp/decimator/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/reshape/module.hh>
#include <jetstream/domains/core/arithmetic/module.hh>
#include <jetstream/domains/core/squeeze_dims/module.hh>
#include <jetstream/domains/core/duplicate/module.hh>

namespace Jetstream::Blocks {

struct DecimatorImpl : public Block::Impl,
                       public DynamicConfig<Blocks::Decimator> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Reshape> reshapeConfig =
        std::make_shared<Modules::Reshape>();
    std::shared_ptr<Modules::Arithmetic> arithmeticConfig =
        std::make_shared<Modules::Arithmetic>();
    std::shared_ptr<Modules::SqueezeDims> squeezeDimsConfig =
        std::make_shared<Modules::SqueezeDims>();
    std::shared_ptr<Modules::Duplicate> duplicateConfig =
        std::make_shared<Modules::Duplicate>();
};

Result DecimatorImpl::validate() {
    const auto& config = *candidate();

    if (config.ratio == 0) {
        JST_ERROR("[BLOCK_DECIMATOR] Ratio must be greater than 0.");
        return Result::ERROR;
    }

    if (axis != config.axis) {
        return Result::RECREATE;
    }

    if (ratio != config.ratio) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result DecimatorImpl::configure() {
    arithmeticConfig->operation = "add";
    arithmeticConfig->axis = axis + 1;
    squeezeDimsConfig->axis = axis + 1;
    duplicateConfig->hostAccessible = true;

    return Result::SUCCESS;
}

Result DecimatorImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer",
                                   "Input",
                                   "Input signal to decimate."));
    JST_CHECK(defineInterfaceOutput("buffer",
                                    "Output",
                                    "Decimated output signal."));

    JST_CHECK(defineInterfaceConfig("axis",
                                    "Axis",
                                    "Axis along which to decimate.",
                                    "int:"));

    JST_CHECK(defineInterfaceConfig("ratio",
                                    "Ratio",
                                    "Decimation ratio.",
                                    "int:"));

    return Result::SUCCESS;
}

Result DecimatorImpl::create() {
    const auto& inputPort = inputs().at("buffer");
    const Tensor& inputTensor = inputPort.tensor;

    // Validate axis against input rank.

    if (axis >= inputTensor.rank()) {
        JST_ERROR("[BLOCK_DECIMATOR] Axis {} is out of bounds for "
                  "input tensor rank {}.", axis, inputTensor.rank());
        return Result::ERROR;
    }

    // Validate axis is divisible by ratio.

    const U64 axisSize = inputTensor.shape(axis);
    if (axisSize % ratio != 0) {
        JST_ERROR("[BLOCK_DECIMATOR] Axis size {} is not divisible "
                  "by ratio {}.", axisSize, ratio);
        return Result::ERROR;
    }

    // Build reshape target shape.
    // e.g. [8192] with axis=0, ratio=4 -> [2048, 4]
    // e.g. [10, 8192] with axis=1, ratio=4 -> [10, 2048, 4]

    const auto& shape = inputTensor.shape();
    std::string shapeStr = "[";
    for (U64 d = 0; d < shape.size(); ++d) {
        if (d > 0) {
            shapeStr += ", ";
        }
        if (d == axis) {
            shapeStr += std::to_string(shape[d] / ratio);
            shapeStr += ", ";
            shapeStr += std::to_string(ratio);
        } else {
            shapeStr += std::to_string(shape[d]);
        }
    }
    shapeStr += "]";

    reshapeConfig->shape = shapeStr;

    // Create reshape module.

    JST_CHECK(moduleCreate("reshape", reshapeConfig, {
        {"buffer", inputPort}
    }));

    // Create arithmetic module (sum along ratio axis).

    JST_CHECK(moduleCreate("arithmetic", arithmeticConfig, {
        {"buffer", moduleGetOutput({"reshape", "buffer"})}
    }));

    // Create squeeze_dims module to remove the reduced axis.

    JST_CHECK(moduleCreate("squeeze_dims", squeezeDimsConfig, {
        {"buffer", moduleGetOutput({"arithmetic", "buffer"})}
    }));

    // Create duplicate module for host accessibility.

    JST_CHECK(moduleCreate("duplicate", duplicateConfig, {
        {"buffer", moduleGetOutput({"squeeze_dims", "buffer"})}
    }));

    JST_CHECK(moduleExposeOutput("buffer",
                                 {"duplicate", "buffer"}));

    auto& outputTensor = outputs()["buffer"].tensor;
    Tensor inputCopy = inputPort.tensor;
    F32 decimationRatio = static_cast<F32>(ratio);

    outputTensor.setDerivedAttribute("sampleRate", [inputCopy, decimationRatio]() -> std::any {
        if (inputCopy.hasAttribute("sampleRate")) {
            return std::any(std::any_cast<F32>(inputCopy.attribute("sampleRate")) / decimationRatio);
        }
        return std::any(0.0f);
    });

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(DecimatorImpl);

}  // namespace Jetstream::Blocks
