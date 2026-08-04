#include <jetstream/domains/dsp/decimator/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/reshape/module.hh>
#include <jetstream/domains/core/arithmetic/module.hh>
#include <jetstream/domains/core/squeeze_dims/module.hh>
#include <jetstream/domains/core/duplicate/module.hh>
#include <jetstream/memory/axis.hh>

#include <optional>
#include <utility>

namespace Jetstream::Blocks {

struct DecimatorImpl : public Block::Impl,
                       public DynamicConfig<Blocks::Decimator> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    struct CandidatePlan {
        std::string reshapeShape;
        I64 childAxis;
        SignalAxes signalAxes;
        SignalAxes reshapedSignalAxes;
        bool deriveSampleRate = false;
    };

    std::optional<CandidatePlan> candidatePlan;
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
    candidatePlan.reset();

    if (config.ratio == 0) {
        JST_ERROR("[BLOCK_DECIMATOR] Ratio must be greater than 0.");
        return Result::ERROR;
    }

    const auto input = inputs().find("buffer");
    if (input != inputs().end() && input->second.resolved()) {
        const Tensor& inputTensor = input->second.tensor;
        SignalAxes axes;
        if (ResolveSignalAxes(inputTensor, axes) != Result::SUCCESS) {
            JST_ERROR("[BLOCK_DECIMATOR] Input signal axis metadata is invalid.");
            return Result::ERROR;
        }
        const Index sampleAxis = *axes.sample;

        const U64 axisSize = inputTensor.shape(sampleAxis);
        if (axisSize % config.ratio != 0) {
            JST_ERROR("[BLOCK_DECIMATOR] Axis size {} is not divisible "
                      "by ratio {}.", axisSize, config.ratio);
            return Result::ERROR;
        }

        CandidatePlan plan;
        plan.reshapeShape = "[";
        for (U64 dimension = 0; dimension < inputTensor.shape().size(); ++dimension) {
            if (dimension > 0) {
                plan.reshapeShape += ", ";
            }
            if (dimension == sampleAxis) {
                plan.reshapeShape += std::to_string(inputTensor.shape(dimension) /
                                                    config.ratio);
                plan.reshapeShape += ", ";
                plan.reshapeShape += std::to_string(config.ratio);
            } else {
                plan.reshapeShape += std::to_string(inputTensor.shape(dimension));
            }
        }
        plan.reshapeShape += "]";

        plan.childAxis = static_cast<I64>(sampleAxis) + 1;
        plan.signalAxes = axes;
        plan.reshapedSignalAxes = axes;
        const auto shiftAfterSample = [sampleAxis](std::optional<Index>& axis) {
            if (axis && *axis > sampleAxis) {
                ++*axis;
            }
        };
        shiftAfterSample(plan.reshapedSignalAxes.batch);
        shiftAfterSample(plan.reshapedSignalAxes.channel);

        if (inputTensor.hasAttribute("sampleRate")) {
            const std::any sampleRate = inputTensor.attribute("sampleRate");
            const auto* sampleRateF32 = std::any_cast<F32>(&sampleRate);
            if (sampleRateF32 == nullptr) {
                JST_ERROR("[BLOCK_DECIMATOR] Sample rate attribute must be F32.");
                return Result::ERROR;
            }
            plan.deriveSampleRate = true;
        }

        candidatePlan = std::move(plan);
    }

    if (ratio != config.ratio) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result DecimatorImpl::configure() {
    arithmeticConfig->operation = "add";
    duplicateConfig->hostAccessible = true;
    duplicateConfig->outputDevice = GetDeviceName(device());

    return Result::SUCCESS;
}

Result DecimatorImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer",
                                   "Input",
                                   "Input signal to decimate."));
    JST_CHECK(defineInterfaceOutput("buffer",
                                    "Output",
                                    "Decimated output signal."));

    JST_CHECK(defineInterfaceConfig("ratio",
                                    "Ratio",
                                    "Decimation ratio.",
                                    "uint:"));

    return Result::SUCCESS;
}

Result DecimatorImpl::create() {
    const auto& inputPort = inputs().at("buffer");
    if (!candidatePlan) {
        JST_ERROR("[BLOCK_DECIMATOR] Input validation plan is unavailable.");
        return Result::ERROR;
    }

    reshapeConfig->shape = candidatePlan->reshapeShape;
    arithmeticConfig->axis = candidatePlan->childAxis;
    squeezeDimsConfig->axis = candidatePlan->childAxis;

    // Create reshape module.

    JST_CHECK(moduleCreate("reshape", reshapeConfig, {
        {"buffer", inputPort}
    }));
    auto reshaped = moduleGetOutput({"reshape", "buffer"});
    JST_CHECK(SetSignalAxes(reshaped.tensor, candidatePlan->reshapedSignalAxes));

    // Create arithmetic module (sum along ratio axis).

    JST_CHECK(moduleCreate("arithmetic", arithmeticConfig, {
        {"buffer", reshaped}
    }));

    // Create squeeze_dims module to remove the reduced axis.

    JST_CHECK(moduleCreate("squeeze_dims", squeezeDimsConfig, {
        {"buffer", moduleGetOutput({"arithmetic", "buffer"})}
    }));
    auto squeezed = moduleGetOutput({"squeeze_dims", "buffer"});
    JST_CHECK(SetSignalAxes(squeezed.tensor, candidatePlan->signalAxes));

    // Create duplicate module for host accessibility.

    JST_CHECK(moduleCreate("duplicate", duplicateConfig, {
        {"buffer", squeezed}
    }));

    JST_CHECK(moduleExposeOutput("buffer",
                                 {"duplicate", "buffer"}));

    auto& outputTensor = outputs()["buffer"].tensor;
    JST_CHECK(SetSignalAxes(outputTensor, candidatePlan->signalAxes));

    if (candidatePlan->deriveSampleRate) {
        const Tensor inputCopy = inputPort.tensor;
        const F32 decimationRatio = static_cast<F32>(ratio);
        JST_CHECK(outputTensor.setDerivedAttribute(
            "sampleRate",
            [inputCopy, decimationRatio]() -> std::any {
                const std::any sampleRate = inputCopy.attribute("sampleRate");
                const auto* sampleRateF32 = std::any_cast<F32>(&sampleRate);
                if (sampleRateF32 == nullptr) {
                    return {};
                }
                return std::any(*sampleRateF32 / decimationRatio);
            }));
    }

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(DecimatorImpl,
                   {"reshape"},
                   {"arithmetic"},
                   {"squeeze_dims"},
                   {"duplicate"});

}  // namespace Jetstream::Blocks
