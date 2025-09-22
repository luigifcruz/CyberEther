#include <jetstream/domains/dsp/rrc_filter/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/rrc_filter/module.hh>

namespace Jetstream::Blocks {

struct RrcFilterImpl : public Block::Impl,
                       public DynamicConfig<Blocks::RrcFilter> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::RrcFilter> filterConfig =
        std::make_shared<Modules::RrcFilter>();
};

Result RrcFilterImpl::configure() {
    filterConfig->symbolRate = symbolRate;
    filterConfig->sampleRate = sampleRate;
    filterConfig->rollOff = rollOff;
    filterConfig->taps = taps;

    return Result::SUCCESS;
}

Result RrcFilterImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer",
                                   "Input",
                                   "Input signal to filter."));

    JST_CHECK(defineInterfaceOutput("buffer",
                                    "Output",
                                    "Filtered output signal."));

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Sample Rate",
                                    "Sampling rate of the input "
                                    "signal.",
                                    "float:MHz:3"));

    JST_CHECK(defineInterfaceConfig("symbolRate",
                                    "Symbol Rate",
                                    "Symbol rate of the modulated "
                                    "signal.",
                                    "float:MHz:3"));

    JST_CHECK(defineInterfaceConfig("rollOff",
                                    "Roll-off Factor",
                                    "Bandwidth-efficiency trade-off "
                                    "(0.0 to 1.0).",
                                    "range:0:1::float"));

    JST_CHECK(defineInterfaceConfig("taps",
                                    "Taps",
                                    "Number of filter coefficients "
                                    "(must be odd, >= 3).",
                                    "int:taps"));

    return Result::SUCCESS;
}

Result RrcFilterImpl::create() {
    JST_CHECK(moduleCreate("filter", filterConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"filter", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(RrcFilterImpl);

}  // namespace Jetstream::Blocks
