#include <jetstream/domains/dsp/phase_correction/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/phase_correction/module.hh>

namespace Jetstream::Blocks {

struct PhaseCorrectionImpl : public Block::Impl,
                             public DynamicConfig<Blocks::PhaseCorrection> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::PhaseCorrection> phaseCorrectionConfig =
        std::make_shared<Modules::PhaseCorrection>();
};

Result PhaseCorrectionImpl::configure() {
    phaseCorrectionConfig->phaseIncrement = phaseIncrement;
    return Result::SUCCESS;
}

Result PhaseCorrectionImpl::define() {
    JST_CHECK(defineInterfaceInput("signal",
                                   "Input",
                                   "Complex input signal."));

    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "Phase-corrected complex signal."));

    JST_CHECK(defineInterfaceConfig("phaseIncrement",
                                    "Phase Increment",
                                    "Phase advance per batch in radians.",
                                    "float:rad:3"));

    return Result::SUCCESS;
}

Result PhaseCorrectionImpl::create() {
    JST_CHECK(moduleCreate("phase_correction", phaseCorrectionConfig, {
        {"signal", inputs().at("signal")}
    }));
    JST_CHECK(moduleExposeOutput("signal", {"phase_correction", "signal"}));
    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(PhaseCorrectionImpl, {"phase_correction"});

}  // namespace Jetstream::Blocks
