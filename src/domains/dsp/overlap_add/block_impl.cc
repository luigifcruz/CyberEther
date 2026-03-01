#include <jetstream/domains/dsp/overlap_add/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/overlap_add/module.hh>

namespace Jetstream::Blocks {

struct OverlapAddImpl : public Block::Impl,
                        public DynamicConfig<Blocks::OverlapAdd> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::OverlapAdd> overlapAddConfig =
        std::make_shared<Modules::OverlapAdd>();
};

Result OverlapAddImpl::configure() {
    overlapAddConfig->axis = axis;

    return Result::SUCCESS;
}

Result OverlapAddImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer",
                                   "Buffer",
                                   "Main input signal."));

    JST_CHECK(defineInterfaceInput("overlap",
                                   "Overlap",
                                   "Overlap region to add."));

    JST_CHECK(defineInterfaceOutput("buffer",
                                    "Output",
                                    "Signal with overlap added."));

    JST_CHECK(defineInterfaceConfig("axis",
                                    "Axis",
                                    "Dimension along which the "
                                    "overlap is applied.",
                                    "int:axis"));

    return Result::SUCCESS;
}

Result OverlapAddImpl::create() {
    JST_CHECK(moduleCreate("overlap_add", overlapAddConfig, {
        {"buffer", inputs().at("buffer")},
        {"overlap", inputs().at("overlap")},
    }));
    JST_CHECK(moduleExposeOutput("buffer",
                                 {"overlap_add", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(OverlapAddImpl);

}  // namespace Jetstream::Blocks
