#include <jetstream/domains/dsp/agc/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/dsp/agc/module.hh>

namespace Jetstream::Blocks {

struct AgcImpl : public Block::Impl, public DynamicConfig<Blocks::Agc> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Agc> moduleConfig = std::make_shared<Modules::Agc>();
};

Result AgcImpl::configure() {
    moduleConfig->tileSize = tileSize;
    moduleConfig->reference = reference;
    moduleConfig->epsilon = epsilon;
    moduleConfig->minGain = minGain;
    moduleConfig->maxGain = maxGain;
    moduleConfig->maxGainChange = maxGainChange;

    return Result::SUCCESS;
}

Result AgcImpl::define() {
    JST_CHECK(defineInterfaceInput("signal", "Input", "Signal to be normalized."));
    JST_CHECK(defineInterfaceOutput("signal", "Output", "Normalized signal."));

    JST_CHECK(defineInterfaceConfig("tileSize",
                                    "Tile Size",
                                    "Samples per RMS estimate; controls response time.",
                                    "uint:samples"));
    JST_CHECK(defineInterfaceConfig("reference",
                                    "Reference",
                                    "Desired RMS output level.",
                                    "float::3"));
    JST_CHECK(defineInterfaceConfig("epsilon",
                                    "Epsilon",
                                    "Power floor added before the square root.",
                                    "float::12"));
    JST_CHECK(defineInterfaceConfig("minGain",
                                    "Min Gain",
                                    "Minimum gain applied to a tile.",
                                    "float::3"));
    JST_CHECK(defineInterfaceConfig("maxGain",
                                    "Max Gain",
                                    "Maximum gain applied to a tile.",
                                    "float::3"));
    JST_CHECK(defineInterfaceConfig("maxGainChange",
                                    "Max Gain Change",
                                    "Maximum gain ratio between adjacent tiles.",
                                    "float::3"));

    return Result::SUCCESS;
}

Result AgcImpl::create() {
    JST_CHECK(moduleCreate("agc", moduleConfig, {
        {"signal", inputs().at("signal")}
    }));
    JST_CHECK(moduleExposeOutput("signal", {"agc", "signal"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(AgcImpl, {"agc"});

}  // namespace Jetstream::Blocks
