#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BASE_HH

#include "bool.hh"
#include "dropdown.hh"
#include "float.hh"
#include "int.hh"
#include "markdown.hh"
#include "multiline.hh"
#include "path.hh"
#include "python.hh"
#include "range.hh"
#include "tensor.hh"
#include "text.hh"
#include "types.hh"
#include "uint.hh"
#include "vector.hh"
#include "vector_inline.hh"

namespace Jetstream {

struct FlowgraphConfigFieldInstance {
    FlowgraphNodeHeightSpec heightSpec() const {
        FlowgraphNodeHeightSpec spec;
        visitFlexible(*this, [&spec](const auto& field) {
            spec = field.heightSpec();
        });
        return spec;
    }

    void update(FlowgraphConfigFieldConfig config) {
        kind = Parser::Get<std::string>(config.format, "type");
        const auto onError = config.onError;
        const auto name = config.name;
        const auto errorId = config.id + "Error";
        unknownFrame.update({
            .id = config.id,
            .label = config.label,
            .help = config.help,
            .background = false,
        });
        Result result = Result::SUCCESS;

        fullHeight = kind == "float" && !Parser::Get<std::string>(config.format, "step_config").empty();

        if (kind == "dropdown") {
            result = dropdown.update(std::move(config));
        } else if (kind == "float") {
            result = floatField.update(std::move(config));
        } else if (kind == "int") {
            result = intField.update(std::move(config));
        } else if (kind == "uint") {
            result = uintField.update(std::move(config));
        } else if (kind == "vector") {
            result = vectorField.update(std::move(config));
        } else if (kind == "vector-inline") {
            result = vectorInline.update(std::move(config));
        } else if (kind == "filepicker" || kind == "filesave") {
            result = path.update(std::move(config));
        } else if (kind == "bool") {
            result = boolField.update(std::move(config));
        } else if (kind == "range") {
            result = range.update(std::move(config));
        } else if (kind == "tensor-config") {
            result = tensor.update(std::move(config));
        } else if (kind == "markdown") {
            result = markdown.update(std::move(config));
        } else if (kind == "python") {
            result = python.update(std::move(config));
        } else if (kind == "multiline") {
            result = multiline.update(std::move(config));
        } else if (kind == "text") {
            result = text.update(std::move(config));
        } else {
            unknownText.update({
                .id = errorId,
                .str = "Unsupported config field: " + kind,
                .tone = Sakura::Text::Tone::Warning,
            });
        }
        if (result != Result::SUCCESS) {
            const auto message = "Invalid configuration value for '" + name + "'.";
            if (onError && notifiedDecodeError != errorId) {
                notifiedDecodeError = errorId;
                onError(result, message);
            }
            kind.clear();
            fullHeight = false;
            unknownText.update({
                .id = errorId,
                .str = message,
                .tone = Sakura::Text::Tone::Warning,
            });
        } else {
            notifiedDecodeError.clear();
        }
    }

    bool isSimple() const {
        if (fullHeight) {
            return false;
        }
        return kind == "dropdown" || kind == "float" || kind == "int" ||
               kind == "uint" || kind == "bool" || kind == "range" ||
               kind == "text" || kind == "vector-inline" ||
               kind == "filepicker" || kind == "filesave";
    }

    void setAllocatedHeight(std::optional<F32> height) {
        visitFlexible(*this, [&height](auto& field) {
            field.setAllocatedHeight(height);
        });
    }

    void render(const Sakura::Context& ctx) const {
        if (kind == "dropdown") {
            dropdown.render(ctx);
        } else if (kind == "float") {
            floatField.render(ctx);
        } else if (kind == "int") {
            intField.render(ctx);
        } else if (kind == "uint") {
            uintField.render(ctx);
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
        } else if (kind == "tensor-config") {
            tensor.render(ctx);
        } else if (kind == "markdown") {
            markdown.render(ctx);
        } else if (kind == "python") {
            python.render(ctx);
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
    template<typename Self, typename Visitor>
    static void visitFlexible(Self& self, Visitor visitor) {
        if (self.kind == "markdown") {
            visitor(self.markdown);
        } else if (self.kind == "python") {
            visitor(self.python);
        }
    }

    std::string kind;
    std::string notifiedDecodeError;
    bool fullHeight = false;
    FlowgraphConfigDropdownField dropdown;
    FlowgraphConfigFloatField floatField;
    FlowgraphConfigIntField intField;
    FlowgraphConfigUIntField uintField;
    FlowgraphConfigVectorField vectorField;
    FlowgraphConfigVectorInlineField vectorInline;
    FlowgraphConfigPathField path;
    FlowgraphConfigBoolField boolField;
    FlowgraphConfigRangeField range;
    FlowgraphConfigTensorField tensor;
    FlowgraphConfigMarkdownField markdown;
    FlowgraphConfigPythonField python;
    FlowgraphConfigMultilineField multiline;
    FlowgraphConfigTextField text;
    Sakura::NodeField unknownFrame;
    Sakura::NodeLabel unknownText;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BASE_HH
