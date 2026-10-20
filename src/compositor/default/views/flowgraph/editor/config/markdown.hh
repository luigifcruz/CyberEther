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
            .title = false,
            .background = false,
            .content = &markdownField,
        });
        markdownField.update({
            .id = this->config.id + "Markdown",
            .value = editing ? buffer : this->config.preview.value_or(value),
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
                    this->config.onApply(std::move(patch), false, {});
                }
                editing = false;
            },
            .onContentOverflow = this->config.onContentOverflow,
        });
        return Result::SUCCESS;
    }

    Sakura::Component& component() {
        return frame;
    }

 private:
    Config config;
    std::string parsedEncoded;
    std::string value;
    bool editing = false;
    std::string buffer;
    Sakura::NodeField frame;
    Sakura::MarkdownEditor markdownField;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MARKDOWN_HH
