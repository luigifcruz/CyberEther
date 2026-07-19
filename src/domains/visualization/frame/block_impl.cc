#include <jetstream/domains/visualization/frame/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/visualization/frame/module.hh>

namespace Jetstream::Blocks {

struct FrameImpl : public Block::Impl, public DynamicConfig<Blocks::Frame> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Frame> frameConfig = std::make_shared<Modules::Frame>();
};

Result FrameImpl::configure() {
    frameConfig->lut = lut;

    return Result::SUCCESS;
}

Result FrameImpl::define() {
    JST_CHECK(defineInterfaceInput("frame", "Frame", "Input F32 frame buffer to display."));

    JST_CHECK(defineInterfaceConfig("lut",
                                    "LUT",
                                    "Apply the Turbo color lookup table to scalar frames.",
                                    "bool"));

    return Result::SUCCESS;
}

Result FrameImpl::create() {
    JST_CHECK(moduleCreate("frame", frameConfig, {
        {"frame", inputs().at("frame")}
    }));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FrameImpl, {"frame"});

}  // namespace Jetstream::Blocks
