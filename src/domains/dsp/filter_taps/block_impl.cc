#include <exception>

#include <jetstream/domains/dsp/filter_taps/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/filter_taps/module.hh>

namespace Jetstream::Blocks {

struct FilterTapsImpl : public Block::Impl, public DynamicConfig<Blocks::FilterTaps> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::FilterTaps> filterTapsConfig =
        std::make_shared<Modules::FilterTaps>();
};

Result FilterTapsImpl::validate() {
    const auto& config = *candidate();

    if (config.heads == 0) {
        JST_ERROR("[BLOCK_FILTER_TAPS] Heads must be greater than 0.");
        return Result::ERROR;
    }

    if (heads != config.heads) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result FilterTapsImpl::configure() {
    if (heads > center.max_size() ||
        heads > filterTapsConfig->center.max_size()) {
        JST_ERROR("[BLOCK_FILTER_TAPS] Heads ({}) exceed the supported configuration size.",
                  heads);
        return Result::ERROR;
    }

    try {
        center.resize(heads);
        filterTapsConfig->center.resize(center.size());
    } catch (const std::exception&) {
        JST_ERROR("[BLOCK_FILTER_TAPS] Failed to allocate configuration for {} heads.",
                  heads);
        return Result::ERROR;
    }

    filterTapsConfig->sampleRate = sampleRate;
    filterTapsConfig->bandwidth = bandwidth;
    for (U64 i = 0; i < center.size(); ++i) {
        filterTapsConfig->center[i] = center[i];
    }
    filterTapsConfig->taps = taps;

    return Result::SUCCESS;
}

Result FilterTapsImpl::define() {
    JST_CHECK(defineInterfaceOutput("coeffs",
                                    "Coefficients",
                                    "FIR bandpass filter coefficients."));

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Sample Rate",
                                    "The sampling rate of the signal.",
                                    "float:MHz:3"));

    JST_CHECK(defineInterfaceConfig("bandwidth",
                                    "Bandwidth",
                                    "The passband bandwidth of the filter.",
                                    "float:MHz:3"));

    JST_CHECK(defineInterfaceConfig("heads",
                                    "Heads",
                                    "Number of filter heads.",
                                    "uint:heads"));

    JST_CHECK(defineInterfaceConfig("center",
                                    "Center",
                                    "The center frequency offset(s) of the filter.",
                                    "vector:float:MHz:3"));

    JST_CHECK(defineInterfaceConfig("taps",
                                    "Taps",
                                    "Number of filter coefficients (must be odd).",
                                    "uint:taps"));

    return Result::SUCCESS;
}

Result FilterTapsImpl::create() {
    JST_CHECK(moduleCreate("filter_taps", filterTapsConfig, {}));
    JST_CHECK(moduleExposeOutput("coeffs", {"filter_taps", "coeffs"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FilterTapsImpl, {"filter_taps"});

}  // namespace Jetstream::Blocks
