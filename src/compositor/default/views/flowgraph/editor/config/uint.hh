#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_UINT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_UINT_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigUIntField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            unit = Parser::Get<std::string>(this->config.format, "unit");
            parsedFormat = this->config.format;
        }
        value = 0;
        JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, value));
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
        });
        input.update({
            .id = this->config.id + "Input",
            .value = value,
            .unit = unit,
            .onChange = [this](U64 nextValue) {
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
    Parser::Map parsedFormat;
    std::string unit;
    U64 value = 0;
    Sakura::NodeField frame;
    Sakura::NodeUIntInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_UINT_HH
