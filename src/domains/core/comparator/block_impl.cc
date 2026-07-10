#include <cmath>

#include <jetstream/domains/core/comparator/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/core/comparator/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

namespace {

constexpr U64 kMaxComparatorInputs = 16;

}  // namespace

struct ComparatorImpl : public Block::Impl, public DynamicConfig<Blocks::Comparator> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Comparator> moduleConfig = std::make_shared<Modules::Comparator>();
    Modules::ComparatorImpl* moduleImpl = nullptr;
};

Result ComparatorImpl::validate() {
    const auto& config = *candidate();

    if (config.inputCount < 2 || config.inputCount > kMaxComparatorInputs) {
        JST_ERROR("[BLOCK_COMPARATOR] Input count must be between 2 and {} (got {}).",
                  kMaxComparatorInputs,
                  config.inputCount);
        return Result::ERROR;
    }

    if (!std::isfinite(config.tolerance) || config.tolerance < 0.0) {
        JST_ERROR("[BLOCK_COMPARATOR] Tolerance must be finite and non-negative (got {}).",
                  config.tolerance);
        return Result::ERROR;
    }

    if (inputCount != config.inputCount) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result ComparatorImpl::configure() {
    moduleConfig->inputCount = inputCount;
    moduleConfig->tolerance = tolerance;

    return Result::SUCCESS;
}

Result ComparatorImpl::define() {
    JST_CHECK(defineInterfaceOutput("error",
                                    "Error",
                                    "Per-element absolute difference against the reference input."));

    for (U64 i = 0; i < inputCount; ++i) {
        const auto index = std::to_string(i);
        const auto label = i == 0 ? std::string("Reference") : ("Input " + index);
        const auto description = i == 0
            ? std::string("Reference tensor other inputs are compared against.")
            : ("Tensor compared against the reference input.");
        JST_CHECK(defineInterfaceInput(jst::fmt::format("input{}", i), label, description));
    }

    JST_CHECK(defineInterfaceConfig("inputCount",
                                    "Input Count",
                                    "Number of tensors to compare (2 to 16).",
                                    "int:"));
    JST_CHECK(defineInterfaceConfig("tolerance",
                                    "Tolerance",
                                    "Maximum allowed absolute difference for a PASS result.",
                                    "float::9"));

    JST_CHECK(defineInterfaceMetric("maxDiff",
                                    "Max Diff",
                                    "Largest absolute difference observed in the latest buffer.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("n/a");
            }
            return jst::fmt::format("{:.6g}", moduleImpl->getMaxDiff());
        }));

    JST_CHECK(defineInterfaceMetric("meanDiff",
                                    "Mean Diff",
                                    "Mean absolute difference observed in the latest buffer.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("n/a");
            }
            return jst::fmt::format("{:.6g}", moduleImpl->getMeanDiff());
        }));

    JST_CHECK(defineInterfaceMetric("mse",
                                    "MSE",
                                    "Mean squared error observed in the latest buffer.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("n/a");
            }
            return jst::fmt::format("{:.6g}", moduleImpl->getMse());
        }));

    JST_CHECK(defineInterfaceMetric("match",
                                    "Match",
                                    "Whether the latest comparison stayed within tolerance.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("N/A");
            }
            return std::string(moduleImpl->getMatch() ? "PASS" : "FAIL");
        }));

    return Result::SUCCESS;
}

Result ComparatorImpl::create() {
    TensorMap moduleInputs;
    for (U64 i = 0; i < inputCount; ++i) {
        const auto port = jst::fmt::format("input{}", i);
        moduleInputs[port] = inputs().at(port);
    }

    JST_CHECK(moduleCreate("comparator", moduleConfig, moduleInputs));
    JST_CHECK(moduleExposeOutput("error", {"comparator", "error"}));

    moduleImpl = moduleHandle("comparator")->getImpl<Modules::ComparatorImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(ComparatorImpl);

}  // namespace Jetstream::Blocks
