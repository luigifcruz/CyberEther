#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MARKDOWN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MARKDOWN_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigMarkdownField {
    using Config = FlowgraphConfigFieldConfig;

    FlowgraphNodeHeightSpec heightSpec() const {
        return {
            .policy = FlowgraphNodeHeightPolicy::FillRemaining,
            .minimum = 120.0f,
        };
    }

    Result update(Config config) {
        this->config = std::move(config);
        std::string nextValue;
        JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, nextValue));
        if (nextValue != parsedEncoded) {
            value = nextValue;
            parsedEncoded = nextValue;
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
        updateEditor();
        return Result::SUCCESS;
    }

    void setAllocatedHeight(std::optional<F32> height) {
        allocatedHeight = height;
        updateEditor();
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            markdownField.render(ctx);
        });
    }

 private:
    void updateEditor() {
        markdownField.update({
            .id = this->config.id + "Markdown",
            .value = editing ? buffer : value,
            .editing = editing,
            .height = allocatedHeight,
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

    Config config;
    std::optional<F32> allocatedHeight;
    std::string parsedEncoded;
    std::string value;
    bool editing = false;
    std::string buffer;
    Sakura::NodeField frame;
    Sakura::Retained::MarkdownEditor markdownField;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MARKDOWN_HH
