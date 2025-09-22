#include <jetstream/domains/visualization/spectrogram/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/visualization/spectrogram/module.hh>

namespace Jetstream::Blocks {

struct SpectrogramImpl : public Block::Impl, public DynamicConfig<Blocks::Spectrogram> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Spectrogram> spectrogramConfig =
        std::make_shared<Modules::Spectrogram>();
};

Result SpectrogramImpl::configure() {
    spectrogramConfig->height = height;

    return Result::SUCCESS;
}

Result SpectrogramImpl::define() {
    JST_CHECK(defineInterfaceInput("signal", "Input",
                                   "Input signal data to visualize."));

    JST_CHECK(defineInterfaceConfig("height",
                                    "Height",
                                    "Number of frequency bins in the vertical axis.",
                                    "int:bins"));

    return Result::SUCCESS;
}

Result SpectrogramImpl::create() {
    JST_CHECK(moduleCreate("spectrogram", spectrogramConfig, {
        {"signal", inputs().at("signal")}
    }));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(SpectrogramImpl);

}  // namespace Jetstream::Blocks
