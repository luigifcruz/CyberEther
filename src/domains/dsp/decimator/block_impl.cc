#include <jetstream/domains/dsp/decimator/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/reshape/module.hh>
#include <jetstream/domains/core/arithmetic/module.hh>
#include <jetstream/domains/core/multiply_constant/module.hh>
#include <jetstream/domains/core/slice/module.hh>
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
        std::string slice;
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
    std::shared_ptr<Modules::MultiplyConstant> normalizeConfig =
        std::make_shared<Modules::MultiplyConstant>();
    std::shared_ptr<Modules::Slice> sliceConfig =
        std::make_shared<Modules::Slice>();
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

    if (config.method != "subsample" &&
        config.method != "sum" &&
        config.method != "average") {
        JST_ERROR("[BLOCK_DECIMATOR] Invalid method '{}'.", config.method);
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
        plan.slice = "[";
        for (U64 dimension = 0; dimension < inputTensor.shape().size(); ++dimension) {
            if (dimension > 0) {
                plan.reshapeShape += ", ";
                plan.slice += ", ";
            }
            if (dimension == sampleAxis) {
                plan.slice += "::" + std::to_string(config.ratio);
                plan.reshapeShape += std::to_string(inputTensor.shape(dimension) /
                                                    config.ratio);
                plan.reshapeShape += ", ";
                plan.reshapeShape += std::to_string(config.ratio);
            } else {
                plan.slice += ":";
                plan.reshapeShape += std::to_string(inputTensor.shape(dimension));
            }
        }
        plan.reshapeShape += "]";
        plan.slice += "]";

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

    if (ratio != config.ratio || method != config.method) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result DecimatorImpl::configure() {
    arithmeticConfig->operation = "add";
    normalizeConfig->constant = 1.0f / static_cast<F32>(ratio);
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
                                    {{"type", "uint"}}));

    JST_CHECK(defineInterfaceConfig("method",
                                    "Method",
                                    "Subsample keeps the first sample in each group; "
                                    "Sum adds the group; Average computes its mean.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "Subsample"}, {"value", "subsample"}},
                                        Parser::Map{{"label", "Sum"}, {"value", "sum"}},
                                        Parser::Map{{"label", "Average"}, {"value", "average"}},
                                    }}}));

    return Result::SUCCESS;
}

Result DecimatorImpl::create() {
    const auto& inputPort = inputs().at("buffer");
    if (!candidatePlan) {
        JST_ERROR("[BLOCK_DECIMATOR] Input validation plan is unavailable.");
        return Result::ERROR;
    }

    auto reduced = inputPort;
    if (method == "subsample") {
        sliceConfig->slice = candidatePlan->slice;
        JST_CHECK(moduleCreate("slice", sliceConfig, {
            {"buffer", inputPort}
        }));
        reduced = moduleGetOutput({"slice", "buffer"});
    } else {
        if (method == "average") {
            // Scale before reduction to avoid overflowing the unnormalized sum.
            JST_CHECK(moduleCreate("normalize", normalizeConfig, {
                {"factor", inputPort}
            }));
            reduced = moduleGetOutput({"normalize", "product"});
        }

        reshapeConfig->shape = candidatePlan->reshapeShape;
        arithmeticConfig->axis = candidatePlan->childAxis;
        squeezeDimsConfig->axis = candidatePlan->childAxis;

        JST_CHECK(moduleCreate("reshape", reshapeConfig, {
            {"buffer", reduced}
        }));
        auto reshaped = moduleGetOutput({"reshape", "buffer"});
        JST_CHECK(SetSignalAxes(reshaped.tensor, candidatePlan->reshapedSignalAxes));

        JST_CHECK(moduleCreate("arithmetic", arithmeticConfig, {
            {"buffer", reshaped}
        }));

        JST_CHECK(moduleCreate("squeeze_dims", squeezeDimsConfig, {
            {"buffer", moduleGetOutput({"arithmetic", "buffer"})}
        }));
        reduced = moduleGetOutput({"squeeze_dims", "buffer"});
    }
    JST_CHECK(SetSignalAxes(reduced.tensor, candidatePlan->signalAxes));

    // Create duplicate module for host accessibility.

    JST_CHECK(moduleCreate("duplicate", duplicateConfig, {
        {"buffer", reduced}
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
                   {"slice"},
                   {"multiply_constant"},
                   {"reshape"},
                   {"arithmetic"},
                   {"squeeze_dims"},
                   {"duplicate"});

}  // namespace Jetstream::Blocks
