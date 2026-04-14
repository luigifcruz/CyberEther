#include <jetstream/domains/dsp/squelch/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/squelch/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

struct SquelchImpl : public Block::Impl, public DynamicConfig<Blocks::Squelch> {
    Result configure() override;
    Result define() override;
    Result create() override;

  protected:
    std::shared_ptr<Modules::Squelch> moduleConfig = std::make_shared<Modules::Squelch>();
    Modules::SquelchImpl* moduleImpl = nullptr;
};

Result SquelchImpl::configure() {
    moduleConfig->threshold = threshold;

    return Result::SUCCESS;
}

Result SquelchImpl::define() {
    JST_CHECK(defineInterfaceInput("signal", "Input", "Signal to be gated by squelch."));
    JST_CHECK(defineInterfaceOutput("signal", "Output", "Input signal passed through while squelch is open."));

    JST_CHECK(defineInterfaceMetric("state",
                                    "Squelch",
                                    "Whether the squelch is currently open or closed.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("Closed");
            }

            return std::string(moduleImpl->getPassing() ? "Open" : "Closed");
        }));

    JST_CHECK(defineInterfaceMetric("amplitude",
                                    "Signal Level",
                                    "How strong the incoming signal is right now.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("0.000");
            }

            return jst::fmt::format("{:.3f}", moduleImpl->getAmplitude());
        }));

    JST_CHECK(defineInterfaceConfig("threshold",
                                    "Threshold",
                                    "Minimum signal level needed to open the squelch.",
                                    "float::3"));

    return Result::SUCCESS;
}

Result SquelchImpl::create() {
    JST_CHECK(moduleCreate("squelch", moduleConfig, {
        {"signal", inputs().at("signal")}
    }));
    JST_CHECK(moduleExposeOutput("signal", {"squelch", "signal"}));

    moduleImpl = moduleHandle("squelch")->getImpl<Modules::SquelchImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(SquelchImpl);

}  // namespace Jetstream::Blocks
