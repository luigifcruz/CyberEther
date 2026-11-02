#include <jetstream/domains/core/cast/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/cast/module.hh>

namespace Jetstream::Blocks {

struct CastImpl : public Block::Impl, public DynamicConfig<Blocks::Cast> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Cast> moduleConfig =
        std::make_shared<Modules::Cast>();
};

Result CastImpl::configure() {
    moduleConfig->outputType = outputType;

    return Result::SUCCESS;
}

Result CastImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer", "Input",
                                   "Input signal to cast."));
    JST_CHECK(defineInterfaceOutput("buffer", "Output",
                                    "Cast output signal."));

    JST_CHECK(defineInterfaceConfig("outputType",
                                    "Output Type",
                                    "The desired output data type.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "CF32"}, {"value", "CF32"}},
                                        Parser::Map{{"label", "F32"}, {"value", "F32"}},
                                        Parser::Map{{"label", "CF64"}, {"value", "CF64"}},
                                        Parser::Map{{"label", "F64"}, {"value", "F64"}},
                                        Parser::Map{{"label", "CI8"}, {"value", "CI8"}},
                                        Parser::Map{{"label", "I8"}, {"value", "I8"}},
                                        Parser::Map{{"label", "CI16"}, {"value", "CI16"}},
                                        Parser::Map{{"label", "I16"}, {"value", "I16"}},
                                        Parser::Map{{"label", "CI32"}, {"value", "CI32"}},
                                        Parser::Map{{"label", "I32"}, {"value", "I32"}},
                                        Parser::Map{{"label", "CI64"}, {"value", "CI64"}},
                                        Parser::Map{{"label", "I64"}, {"value", "I64"}},
                                        Parser::Map{{"label", "CU8"}, {"value", "CU8"}},
                                        Parser::Map{{"label", "U8"}, {"value", "U8"}},
                                        Parser::Map{{"label", "CU16"}, {"value", "CU16"}},
                                        Parser::Map{{"label", "U16"}, {"value", "U16"}},
                                        Parser::Map{{"label", "CU32"}, {"value", "CU32"}},
                                        Parser::Map{{"label", "U32"}, {"value", "U32"}},
                                        Parser::Map{{"label", "CU64"}, {"value", "CU64"}},
                                        Parser::Map{{"label", "U64"}, {"value", "U64"}},
                                    }}}));

    return Result::SUCCESS;
}

Result CastImpl::create() {
    JST_CHECK(moduleCreate("cast", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"cast", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(CastImpl, {"cast"});

}  // namespace Jetstream::Blocks
