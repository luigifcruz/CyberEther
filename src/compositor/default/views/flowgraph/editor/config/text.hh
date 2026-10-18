#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TEXT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TEXT_HH

#include "types.hh"

#include "../field_tooltip.hh"

namespace Jetstream {

struct FlowgraphConfigTextField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config) {
        this->config = std::move(config);
        value.clear();
        JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, value));
        help.update(this->config.id, this->config.help);
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .tooltip = help.overlay(),
            .content = &input,
        });
        input.update({
            .id = this->config.id + "Input",
            .value = value,
            .submit = Sakura::NodeTextInput::Submit::OnEnter,
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

    Sakura::Component& component() {
        return frame;
    }

 private:
    Config config;
    std::string value;
    FlowgraphFieldTooltip help;
    Sakura::NodeField frame;
    Sakura::NodeTextInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TEXT_HH
