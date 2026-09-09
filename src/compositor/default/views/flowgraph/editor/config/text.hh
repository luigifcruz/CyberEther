#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TEXT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TEXT_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigTextField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config) {
        this->config = std::move(config);
        value.clear();
        JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, value));
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
        });
        input.update({
            .id = this->config.id + "Input",
            .value = value,
            .submit = Sakura::TextInput::Submit::OnEnter,
            .onChange = [this](const std::string& nextValue) {
                Parser::Map patch;
                patch[this->config.name] = nextValue;
                if (this->config.onApply) {
                    this->config.onApply(std::move(patch), false);
                }
            },
        });
        return Result::SUCCESS;
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            input.render(ctx);
        });
    }

 private:
    Config config;
    std::string value;
    Sakura::NodeField frame;
    Sakura::NodeTextInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TEXT_HH
