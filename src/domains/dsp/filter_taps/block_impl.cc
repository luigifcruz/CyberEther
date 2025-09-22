#include <jetstream/domains/dsp/filter_taps/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <algorithm>

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
    const U64 effectiveHeads = std::max(config.heads,
                                        static_cast<U64>(config.center.size()));

    if (effectiveHeads == 0) {
        JST_ERROR("[BLOCK_FILTER_TAPS] Heads must be greater than 0.");
        return Result::ERROR;
    }

    if (heads != effectiveHeads) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result FilterTapsImpl::configure() {
    // Keep legacy center-only configs working by treating center length as
    // authoritative when it exceeds heads.
    heads = std::max(heads, static_cast<U64>(center.size()));
    center.resize(heads);

    filterTapsConfig->sampleRate = sampleRate;
    filterTapsConfig->bandwidth = bandwidth;
    filterTapsConfig->center.resize(center.size());
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
                                    "int:heads"));

    JST_CHECK(defineInterfaceConfig("center",
                                    "Center",
                                    "The center frequency offset(s) of the filter.",
                                    "vector:float:MHz:3"));

    JST_CHECK(defineInterfaceConfig("taps",
                                    "Taps",
                                    "Number of filter coefficients (must be odd).",
                                    "int:taps"));

    return Result::SUCCESS;
}

Result FilterTapsImpl::create() {
    JST_CHECK(moduleCreate("filter_taps", filterTapsConfig, {}));
    JST_CHECK(moduleExposeOutput("coeffs", {"filter_taps", "coeffs"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FilterTapsImpl);

}  // namespace Jetstream::Blocks
