#include <jetstream/domains/core/complex_merge/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/complex_merge/module.hh>

namespace Jetstream::Blocks {

struct ComplexMergeImpl : public Block::Impl, public DynamicConfig<Blocks::ComplexMerge> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::ComplexMerge> moduleConfig =
        std::make_shared<Modules::ComplexMerge>();
};

Result ComplexMergeImpl::configure() {
    moduleConfig->mode = mode;

    return Result::SUCCESS;
}

Result ComplexMergeImpl::define() {
    const bool polar = candidate()->mode == "polar";

    JST_CHECK(defineInterfaceInput("a", polar ? "Magnitude" : "Real",
                                   polar ? "Magnitude of each sample." :
                                           "Real part of each sample."));
    JST_CHECK(defineInterfaceInput("b", polar ? "Phase" : "Imag",
                                   polar ? "Phase of each sample in radians." :
                                           "Imaginary part of each sample."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output",
                                    "Merged complex signal."));

    JST_CHECK(defineInterfaceConfig("mode",
                                    "Mode",
                                    "Parts the complex signal is built from.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "Real and Imag"}, {"value", "rectangular"}},
                                        Parser::Map{{"label", "Magnitude and Phase"}, {"value", "polar"}},
                                    }}}));

    return Result::SUCCESS;
}

Result ComplexMergeImpl::create() {
    JST_CHECK(moduleCreate("complex_merge", moduleConfig, {
        {"a", inputs().at("a")},
        {"b", inputs().at("b")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"complex_merge", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(ComplexMergeImpl, {"complex_merge"});

}  // namespace Jetstream::Blocks
