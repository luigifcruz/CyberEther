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
    frameConfig->fit = fit;
    frameConfig->colormap = colormap;
    frameConfig->autoRange = autoRange;
    frameConfig->smooth = smooth;
    frameConfig->xLabel = xLabel;
    frameConfig->yLabel = yLabel;

    return Result::SUCCESS;
}

Result FrameImpl::define() {
    JST_CHECK(defineInterfaceInput("frame", "Frame", "Input F32 frame buffer to display."));

    JST_CHECK(defineInterfaceConfig("fit",
                                    "Fit",
                                    "How the frame fills the surface.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "Contain"}, {"value", "contain"}},
                                        Parser::Map{{"label", "Cover"}, {"value", "cover"}},
                                        Parser::Map{{"label", "Stretch"}, {"value", "stretch"}},
                                    }}}));

    const auto input = inputs().find("frame");
    if (input != inputs().end() && input->second.resolved()) {
        const Tensor& frame = input->second.tensor;
        if (frame.rank() == 2 || (frame.rank() == 3 && frame.shape(2) == 1)) {
            JST_CHECK(defineInterfaceConfig("colormap",
                                            "Colormap",
                                            "Color lookup applied to scalar frames.",
                                            {{"type", "dropdown"}, {"options", Parser::Sequence{
                                                Parser::Map{{"label", "Grayscale"}, {"value", "grayscale"}},
                                                Parser::Map{{"label", "Turbo"}, {"value", "turbo"}},
                                                Parser::Map{{"label", "Viridis"}, {"value", "viridis"}},
                                                Parser::Map{{"label", "Inferno"}, {"value", "inferno"}},
                                                Parser::Map{{"label", "Magma"}, {"value", "magma"}},
                                                Parser::Map{{"label", "Plasma"}, {"value", "plasma"}},
                                            }}}));
        }
    }

    JST_CHECK(defineInterfaceConfig("autoRange",
                                    "Auto Range",
                                    "Map the observed minimum and maximum of each frame to the display range. "
                                    "When off, values are shown as is in the 0 to 1 range.",
                                    {{"type", "bool"}}));

    JST_CHECK(defineInterfaceConfig("smooth",
                                    "Smooth",
                                    "Sample the frame with bilinear interpolation.",
                                    {{"type", "bool"}}));

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
