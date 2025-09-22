#include <jetstream/domains/dsp/window/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/window/module.hh>

namespace Jetstream::Blocks {

struct WindowImpl : public Block::Impl, public DynamicConfig<Blocks::Window> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Window> windowConfig = std::make_shared<Modules::Window>();
};

Result WindowImpl::configure() {
    windowConfig->size = size;

    return Result::SUCCESS;
}

Result WindowImpl::define() {
    JST_CHECK(defineInterfaceOutput("window",
                                    "Window",
                                    "Generated Blackman window coefficients."));

    JST_CHECK(defineInterfaceConfig("size",
                                    "Size",
                                    "Number of samples in the window.",
                                    "int:samples"));

    return Result::SUCCESS;
}

Result WindowImpl::create() {
    JST_CHECK(moduleCreate("window", windowConfig, {}));
    JST_CHECK(moduleExposeOutput("window", {"window", "window"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(WindowImpl);

}  // namespace Jetstream::Blocks
