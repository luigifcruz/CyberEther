#include <jetstream/domains/core/complex_split/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/complex_split/module.hh>

namespace Jetstream::Blocks {

struct ComplexSplitImpl : public Block::Impl, public DynamicConfig<Blocks::ComplexSplit> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::ComplexSplit> moduleConfig =
        std::make_shared<Modules::ComplexSplit>();
};

Result ComplexSplitImpl::configure() {
    moduleConfig->mode = mode;

    return Result::SUCCESS;
}

Result ComplexSplitImpl::define() {
    const bool polar = candidate()->mode == "polar";

    JST_CHECK(defineInterfaceInput("buffer", "Input",
                                   "Complex signal to split."));
    JST_CHECK(defineInterfaceOutput("a", polar ? "Magnitude" : "Real",
                                    polar ? "Magnitude of each sample." :
                                            "Real part of each sample."));
    JST_CHECK(defineInterfaceOutput("b", polar ? "Phase" : "Imag",
                                    polar ? "Phase of each sample in radians." :
                                            "Imaginary part of each sample."));

    JST_CHECK(defineInterfaceConfig("mode",
                                    "Mode",
                                    "Parts the complex signal is split into.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "Real and Imag"}, {"value", "rectangular"}},
                                        Parser::Map{{"label", "Magnitude and Phase"}, {"value", "polar"}},
                                    }}}));

    return Result::SUCCESS;
}

Result ComplexSplitImpl::create() {
    JST_CHECK(moduleCreate("complex_split", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("a", {"complex_split", "a"}));
    JST_CHECK(moduleExposeOutput("b", {"complex_split", "b"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(ComplexSplitImpl, {"complex_split"});

}  // namespace Jetstream::Blocks
