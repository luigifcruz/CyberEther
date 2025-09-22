#include <jetstream/domains/core/add/block.hh>
#include "jetstream/detail/block_impl.hh"

#include <jetstream/domains/core/add/module.hh>

namespace Jetstream::Blocks {

struct AddImpl : public Block::Impl, public DynamicConfig<Blocks::Add> {
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Add> moduleConfig = std::make_shared<Modules::Add>();
};

Result AddImpl::define() {
    JST_CHECK(defineInterfaceOutput("sum", "Output", "Sum of the two input signals."));

    JST_CHECK(defineInterfaceInput("a", "Input A", "First signal to be added."));
    JST_CHECK(defineInterfaceInput("b", "Input B", "Second signal to be added."));

    return Result::SUCCESS;
}

Result AddImpl::create() {
    JST_CHECK(moduleCreate("add", moduleConfig, {
        {"a", inputs().at("a")},
        {"b", inputs().at("b")}
    }));
    JST_CHECK(moduleExposeOutput("sum", {"add", "sum"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(AddImpl);

}  // namespace Jetstream::Blocks
