#include <jetstream/domains/core/arithmetic/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/arithmetic/module.hh>

namespace Jetstream::Blocks {

struct ArithmeticImpl : public Block::Impl,
                        public DynamicConfig<Blocks::Arithmetic> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Arithmetic> moduleConfig =
        std::make_shared<Modules::Arithmetic>();
};

Result ArithmeticImpl::configure() {
    moduleConfig->operation = operation;
    moduleConfig->axis = axis;
    moduleConfig->squeeze = squeeze;

    return Result::SUCCESS;
}

Result ArithmeticImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer",
                                   "Input",
                                   "Input signal to reduce."));
    JST_CHECK(defineInterfaceOutput("buffer",
                                    "Output",
                                    "Reduced output signal."));

    JST_CHECK(defineInterfaceConfig("operation",
                                    "Operation",
                                    "Arithmetic operation to apply.",
                                    "dropdown:add(Add),sub(Subtract),"
                                    "mul(Multiply),div(Divide)"));

    JST_CHECK(defineInterfaceConfig("axis",
                                    "Axis",
                                    "Axis along which to reduce.",
                                    "int:"));

    JST_CHECK(defineInterfaceConfig("squeeze",
                                    "Squeeze",
                                    "Remove the reduced dimension.",
                                    "bool"));

    return Result::SUCCESS;
}

Result ArithmeticImpl::create() {
    JST_CHECK(moduleCreate("arithmetic", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));
    JST_CHECK(moduleExposeOutput("buffer", {"arithmetic", "buffer"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(ArithmeticImpl);

}  // namespace Jetstream::Blocks
