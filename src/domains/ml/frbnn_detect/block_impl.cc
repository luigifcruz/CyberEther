#include <jetstream/domains/ml/frbnn_detect/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/ml/frbnn_detect/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

struct FrbnnDetectImpl : public Block::Impl, public DynamicConfig<Blocks::FrbnnDetect> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::FrbnnDetect> moduleConfig = std::make_shared<Modules::FrbnnDetect>();
    Modules::FrbnnDetectImpl* moduleImpl = nullptr;
};

Result FrbnnDetectImpl::configure() {
    moduleConfig->threshold  = threshold;
    moduleConfig->classIndex = classIndex;
    return Result::SUCCESS;
}

Result FrbnnDetectImpl::define() {
    JST_CHECK(defineInterfaceInput("probabilities",
                                   "Probabilities",
                                   "F32 tensor [batch] or [batch, n_classes] from the Infer block."));

    JST_CHECK(defineInterfaceOutput("signal",
                                    "Signal",
                                    "F32 tensor [batch] — FRB-class probability per sample."));

    JST_CHECK(defineInterfaceConfig("threshold",
                                    "Threshold",
                                    "Probability above which a sample is flagged as an FRB candidate.",
                                    "slider:0:1:0.01"));

    JST_CHECK(defineInterfaceConfig("classIndex",
                                    "Class Index",
                                    "Column of the model output that represents the FRB class.",
                                    "int:index"));

    JST_CHECK(defineInterfaceMetric("totalCandidates",
                                    "Candidates",
                                    "Total FRB candidates detected since start.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("0");
            }
            return jst::fmt::format("{}", moduleImpl->getTotalCandidates());
        }));

    JST_CHECK(defineInterfaceMetric("latestProbability",
                                    "Max P(FRB)",
                                    "Maximum FRB probability in the latest batch.",
                                    "progressbar",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::pair<std::string, F32>{"0.0%", 0.0f};
            }
            const F32 p = moduleImpl->getLatestProbability();
            return std::pair<std::string, F32>{jst::fmt::format("{:.1f}%", p * 100.0f), p};
        }));

    return Result::SUCCESS;
}

Result FrbnnDetectImpl::create() {
    JST_CHECK(moduleCreate("frbnn_detect", moduleConfig, {
        {"probabilities", inputs().at("probabilities")},
    }));
    JST_CHECK(moduleExposeOutput("signal", {"frbnn_detect", "signal"}));

    moduleImpl = moduleHandle("frbnn_detect")->getImpl<Modules::FrbnnDetectImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FrbnnDetectImpl);

}  // namespace Jetstream::Blocks
