#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BASE_HH

#include "bool.hh"
#include "dropdown.hh"
#include "float.hh"
#include "int.hh"
#include "markdown.hh"
#include "multiline.hh"
#include "path.hh"
#include "range.hh"
#include "text.hh"
#include "types.hh"
#include "vector.hh"
#include "vector_inline.hh"

namespace Jetstream {

struct FlowgraphConfigFieldInstance : public Sakura::Component {
    void update(FlowgraphConfigFieldConfig config) {
        const auto parts = Parser::SplitString(config.format, ":");
        kind = parts.empty() ? "" : parts[0];

        if (kind == "dropdown") {
            dropdown.update(std::move(config));
        } else if (kind == "float") {
            floatField.update(std::move(config));
        } else if (kind == "int") {
            intField.update(std::move(config));
        } else if (kind == "vector") {
            vectorField.update(std::move(config));
        } else if (kind == "vector-inline") {
            vectorInline.update(std::move(config));
        } else if (kind == "filepicker" || kind == "filesave") {
            path.update(std::move(config));
        } else if (kind == "bool") {
            boolField.update(std::move(config));
        } else if (kind == "range") {
            range.update(std::move(config));
        } else if (kind == "markdown") {
            markdown.update(std::move(config));
        } else if (kind == "multiline") {
            multiline.update(std::move(config));
        } else if (kind == "text") {
            text.update(std::move(config));
        } else {
            unknownFrame.update({
                .id = config.id,
                .label = config.label,
                .help = config.help,
                .background = false,
            });
            unknownText.update({
                .id = config.id + "Unsupported",
                .str = "Unsupported config field: " + kind,
                .tone = Sakura::Text::Tone::Warning,
            });
        }
    }

    void render(const Sakura::Context& ctx) const {
        if (kind == "dropdown") {
            dropdown.render(ctx);
        } else if (kind == "float") {
            floatField.render(ctx);
        } else if (kind == "int") {
            intField.render(ctx);
        } else if (kind == "vector") {
            vectorField.render(ctx);
        } else if (kind == "vector-inline") {
            vectorInline.render(ctx);
        } else if (kind == "filepicker" || kind == "filesave") {
            path.render(ctx);
        } else if (kind == "bool") {
            boolField.render(ctx);
        } else if (kind == "range") {
            range.render(ctx);
        } else if (kind == "markdown") {
            markdown.render(ctx);
        } else if (kind == "multiline") {
            multiline.render(ctx);
        } else if (kind == "text") {
            text.render(ctx);
        } else {
            unknownFrame.render(ctx, [this](const Sakura::Context& ctx) {
                unknownText.render(ctx);
            });
        }
    }

 private:
    std::string kind;
    FlowgraphConfigDropdownField dropdown;
    FlowgraphConfigFloatField floatField;
    FlowgraphConfigIntField intField;
    FlowgraphConfigVectorField vectorField;
    FlowgraphConfigVectorInlineField vectorInline;
    FlowgraphConfigPathField path;
    FlowgraphConfigBoolField boolField;
    FlowgraphConfigRangeField range;
    FlowgraphConfigMarkdownField markdown;
    FlowgraphConfigMultilineField multiline;
    FlowgraphConfigTextField text;
    Sakura::NodeField unknownFrame;
    Sakura::NodeLabel unknownText;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BASE_HH
