#include <jetstream/domains/dsp/fold/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/fold/module.hh>

namespace Jetstream::Blocks {

struct FoldImpl : public Block::Impl, public DynamicConfig<Blocks::Fold> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Fold> foldConfig =
        std::make_shared<Modules::Fold>();
};

Result FoldImpl::configure() {
    foldConfig->axis = axis;
    foldConfig->offset = offset;
    foldConfig->size = size;

    return Result::SUCCESS;
}

Result FoldImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer",
                                   "Input",
                                   "Input signal to fold."));

    JST_CHECK(defineInterfaceOutput("buffer",
                                    "Output",
                                    "Folded output signal."));

    JST_CHECK(defineInterfaceConfig("axis",
                                    "Axis",
                                    "Dimension along which to fold.",
                                    "int:axis"));

    JST_CHECK(defineInterfaceConfig("offset",
                                    "Offset",
                                    "Sample offset before folding.",
                                    "int:samples"));

    JST_CHECK(defineInterfaceConfig("size",
                                    "Size",
                                    "Output size along the folded axis.",
                                    "int:samples"));

    return Result::SUCCESS;
}

Result FoldImpl::create() {
    JST_CHECK(moduleCreate("fold", foldConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"fold", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(FoldImpl);

}  // namespace Jetstream::Blocks
