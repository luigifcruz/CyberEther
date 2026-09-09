#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BOOL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BOOL_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigBoolField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config) {
        this->config = std::move(config);
        value = false;
        JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, value));
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
        });
        input.update({
            .id = this->config.id + "Input",
            .value = value,
            .onChange = [this](bool nextValue) {
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
    bool value = false;
    Sakura::NodeField frame;
    Sakura::NodeBoolInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BOOL_HH
