#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TEXT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TEXT_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigTextField : public Sakura::Component {
    using Config = FlowgraphConfigFieldConfig;

    void update(Config config) {
        this->config = std::move(config);
        if (this->config.encoded != parsedEncoded) {
            value = this->config.encoded;
            parsedEncoded = this->config.encoded;
        }
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
                auto values = this->config.values;
                values[this->config.name] = nextValue;
                if (this->config.onApply) {
                    this->config.onApply(std::move(values), false);
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            input.render(ctx);
        });
    }

 private:
    Config config;
    std::string parsedEncoded;
    std::string value;
    Sakura::NodeField frame;
    Sakura::NodeTextInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TEXT_HH
