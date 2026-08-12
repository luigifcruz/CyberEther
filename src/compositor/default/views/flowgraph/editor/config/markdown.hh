#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MARKDOWN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MARKDOWN_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigMarkdownField {
    using Config = FlowgraphConfigFieldConfig;

    void update(Config config) {
        this->config = std::move(config);
        if (this->config.encoded != parsedEncoded) {
            value = this->config.encoded;
            parsedEncoded = this->config.encoded;
            if (!editing) {
                buffer = value;
            }
        }
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
            .title = false,
            .background = false,
        });
        markdownField.update({
            .id = this->config.id + "Markdown",
            .value = editing ? buffer : value,
            .editing = editing,
            .onChange = [this](std::string nextValue) {
                buffer = std::move(nextValue);
            },
            .onEdit = [this]() {
                buffer = value;
                editing = true;
            },
            .onDone = [this]() {
                Parser::Map patch;
                patch[this->config.name] = buffer;
                if (this->config.onApply) {
                    this->config.onApply(std::move(patch), false);
                }
                editing = false;
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            markdownField.render(ctx);
        });
    }

 private:
    Config config;
    std::string parsedEncoded;
    std::string value;
    bool editing = false;
    std::string buffer;
    Sakura::NodeField frame;
    Sakura::Retained::MarkdownEditor markdownField;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MARKDOWN_HH
