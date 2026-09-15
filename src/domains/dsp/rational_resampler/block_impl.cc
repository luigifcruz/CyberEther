#include <jetstream/domains/dsp/rational_resampler/block.hh>
#include <jetstream/domains/dsp/rational_resampler/module.hh>
#include <jetstream/detail/block_impl.hh>

namespace Jetstream::Blocks {

struct RationalResamplerImpl : public Block::Impl,
                               public DynamicConfig<Blocks::RationalResampler> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::RationalResampler> resamplerConfig =
        std::make_shared<Modules::RationalResampler>();
};

Result RationalResamplerImpl::configure() {
    resamplerConfig->interpolation = interpolation;
    resamplerConfig->decimation = decimation;
    resamplerConfig->taps = taps;
    resamplerConfig->cutoff = cutoff;
    return Result::SUCCESS;
}

Result RationalResamplerImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input", "Signal to resample."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output", "Resampled signal."));

    JST_CHECK(defineInterfaceConfig("interpolation", "Interpolation",
                                    "Positive numerator of the sample-rate ratio.",
                                    {{"type", "uint"}}));
    JST_CHECK(defineInterfaceConfig("decimation", "Decimation",
                                    "Positive denominator of the sample-rate ratio.",
                                    {{"type", "uint"}}));
    JST_CHECK(defineInterfaceConfig("taps", "Taps",
                                    "Odd FIR length at the interpolated rate. "
                                    "Zero selects an automatic length.",
                                    {{"type", "uint"}, {"unit", "taps"}}));
    JST_CHECK(defineInterfaceConfig("cutoff", "Cutoff",
                                    "Fraction of the smaller Nyquist frequency "
                                    "(strictly between 0 and 1).",
                                    {{"type", "float"}, {"precision", 3}}));
    return Result::SUCCESS;
}

Result RationalResamplerImpl::create() {
    JST_CHECK(moduleCreate("resampler", resamplerConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"resampler", "buffer"}));
    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(RationalResamplerImpl, {"rational_resampler"});

}  // namespace Jetstream::Blocks
