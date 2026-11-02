#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BASE_HH

#include "bool.hh"
#include "console.hh"
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

#include "../field_tooltip.hh"

#include <memory>
#include <unordered_map>
#include <vector>

namespace Jetstream {

struct FlowgraphConfigFieldInstance {
    enum class Kind {
        Unknown,
        Dropdown,
        Float,
        Int,
        UInt,
        Vector,
        VectorInline,
        Path,
        Bool,
        Range,
        Tensor,
        Markdown,
        Python,
        PythonConsole,
        Multiline,
        Text,
    };

    FlowgraphNodeHeightSpec heightSpec() const {
        if (failed) {
            return {};
        }
        if (kind == Kind::Markdown && markdown) {
            return markdown->heightSpec();
        }
        if (kind == Kind::Python && python) {
            return python->heightSpec();
        }
        if (kind == Kind::PythonConsole && pythonConsole) {
            return pythonConsole->heightSpec();
        }
        return {};
    }

    void update(FlowgraphConfigFieldConfig config) {
        const std::string nextKind = Parser::Get<std::string>(config.format, "type");
        std::vector<std::shared_ptr<void>> replaced;
        if (nextKind != builtKind || config.id != builtId) {
            replaced = {std::move(dropdown), std::move(floatField), std::move(intField), std::move(uintField),
                        std::move(vectorField), std::move(vectorInline), std::move(path), std::move(boolField),
                        std::move(range), std::move(tensor), std::move(markdown), std::move(python),
                        std::move(pythonConsole), std::move(multiline), std::move(text)};
            reset();
            builtId = config.id;
            builtKind = nextKind;
            kind = ParseKind(nextKind);
        }
        failed = false;
        const auto onError = config.onError;
        const auto name = config.name;
        const auto errorId = config.id + "Error";
        unknownHelp.update(config.id, config.help);
        unknownFrame.update({
            .id = config.id,
            .label = config.label,
            .tooltip = unknownHelp.overlay(),
            .background = false,
            .content = &unknownText,
        });
        Result result = Result::SUCCESS;

        fullHeight = kind == Kind::Float && !Parser::Get<std::string>(config.format, "step_config").empty();

        switch (kind) {
            case Kind::Dropdown:
                result = ensure(dropdown).update(std::move(config));
                break;
            case Kind::Float:
                result = ensure(floatField).update(std::move(config));
                break;
            case Kind::Int:
                result = ensure(intField).update(std::move(config));
                break;
            case Kind::UInt:
                result = ensure(uintField).update(std::move(config));
                break;
            case Kind::Vector:
                result = ensure(vectorField).update(std::move(config));
                break;
            case Kind::VectorInline:
                result = ensure(vectorInline).update(std::move(config));
                break;
            case Kind::Path:
                result = ensure(path).update(std::move(config));
                break;
            case Kind::Bool:
                result = ensure(boolField).update(std::move(config));
                break;
            case Kind::Range:
                result = ensure(range).update(std::move(config));
                break;
            case Kind::Tensor:
                result = ensure(tensor).update(std::move(config));
                break;
            case Kind::Markdown:
                result = ensure(markdown).update(std::move(config));
                break;
            case Kind::Python:
                result = ensure(python).update(std::move(config));
                break;
            case Kind::PythonConsole:
                result = ensure(pythonConsole).update(std::move(config));
                break;
            case Kind::Multiline:
                result = ensure(multiline).update(std::move(config));
                break;
            case Kind::Text:
                result = ensure(text).update(std::move(config));
                break;
            case Kind::Unknown:
                unknownText.update({
                    .id = errorId,
                    .str = "Unsupported config field: " + builtKind,
                    .colorKey = "warning_yellow",
                });
                break;
        }
        if (result != Result::SUCCESS) {
            const auto message = jst::fmt::format("Invalid configuration value for '{}'.", name);
            if (onError && notifiedDecodeError != errorId) {
                notifiedDecodeError = errorId;
                onError(result, message);
            }
            failed = true;
            unknownText.update({
                .id = errorId,
                .str = message,
                .colorKey = "warning_yellow",
            });
        } else {
            notifiedDecodeError.clear();
        }
    }

    bool isSimple() const {
        if (fullHeight) {
            return false;
        }
        switch (kind) {
            case Kind::Dropdown:
            case Kind::Float:
            case Kind::Int:
            case Kind::UInt:
            case Kind::Bool:
            case Kind::Range:
            case Kind::Text:
            case Kind::VectorInline:
            case Kind::Path:
                return true;
            default:
                return false;
        }
    }

    Sakura::Component& component() {
        if (failed) {
            return unknownFrame;
        }
        switch (kind) {
            case Kind::Dropdown:
                return dropdown ? dropdown->component() : unknownFrame;
            case Kind::Float:
                return floatField ? floatField->component() : unknownFrame;
            case Kind::Int:
                return intField ? intField->component() : unknownFrame;
            case Kind::UInt:
                return uintField ? uintField->component() : unknownFrame;
            case Kind::Vector:
                return vectorField ? vectorField->component() : unknownFrame;
            case Kind::VectorInline:
                return vectorInline ? vectorInline->component() : unknownFrame;
            case Kind::Path:
                return path ? path->component() : unknownFrame;
            case Kind::Bool:
                return boolField ? boolField->component() : unknownFrame;
            case Kind::Range:
                return range ? range->component() : unknownFrame;
            case Kind::Tensor:
                return tensor ? tensor->component() : unknownFrame;
            case Kind::Markdown:
                return markdown ? markdown->component() : unknownFrame;
            case Kind::Python:
                return python ? python->component() : unknownFrame;
            case Kind::PythonConsole:
                return pythonConsole ? pythonConsole->component() : unknownFrame;
            case Kind::Multiline:
                return multiline ? multiline->component() : unknownFrame;
            case Kind::Text:
                return text ? text->component() : unknownFrame;
            case Kind::Unknown:
                return unknownFrame;
        }
        return unknownFrame;
    }

 private:
    static Kind ParseKind(const std::string& value) {
        static const std::unordered_map<std::string, Kind> kinds = {
            {"dropdown", Kind::Dropdown},
            {"float", Kind::Float},
            {"int", Kind::Int},
            {"uint", Kind::UInt},
            {"vector", Kind::Vector},
            {"vector-inline", Kind::VectorInline},
            {"filepicker", Kind::Path},
            {"filesave", Kind::Path},
            {"bool", Kind::Bool},
            {"range", Kind::Range},
            {"tensor-config", Kind::Tensor},
            {"markdown", Kind::Markdown},
            {"python", Kind::Python},
            {"python-console", Kind::PythonConsole},
            {"multiline", Kind::Multiline},
            {"text", Kind::Text},
        };
        const auto it = kinds.find(value);
        return it == kinds.end() ? Kind::Unknown : it->second;
    }

    template<typename T>
    static T& ensure(std::unique_ptr<T>& field) {
        if (!field) {
            field = std::make_unique<T>();
        }
        return *field;
    }

    void reset() {
        dropdown.reset();
        floatField.reset();
        intField.reset();
        uintField.reset();
        vectorField.reset();
        vectorInline.reset();
        path.reset();
        boolField.reset();
        range.reset();
        tensor.reset();
        markdown.reset();
        python.reset();
        pythonConsole.reset();
        multiline.reset();
        text.reset();
    }

    Kind kind = Kind::Unknown;
    bool failed = false;
    std::string builtKind;
    std::string builtId;
    std::string notifiedDecodeError;
    bool fullHeight = false;
    std::unique_ptr<FlowgraphConfigDropdownField> dropdown;
    std::unique_ptr<FlowgraphConfigFloatField> floatField;
    std::unique_ptr<FlowgraphConfigIntField> intField;
    std::unique_ptr<FlowgraphConfigUIntField> uintField;
    std::unique_ptr<FlowgraphConfigVectorField> vectorField;
    std::unique_ptr<FlowgraphConfigVectorInlineField> vectorInline;
    std::unique_ptr<FlowgraphConfigPathField> path;
    std::unique_ptr<FlowgraphConfigBoolField> boolField;
    std::unique_ptr<FlowgraphConfigRangeField> range;
    std::unique_ptr<FlowgraphConfigTensorField> tensor;
    std::unique_ptr<FlowgraphConfigMarkdownField> markdown;
    std::unique_ptr<FlowgraphConfigPythonField> python;
    std::unique_ptr<FlowgraphConfigPythonConsoleField> pythonConsole;
    std::unique_ptr<FlowgraphConfigMultilineField> multiline;
    std::unique_ptr<FlowgraphConfigTextField> text;
    FlowgraphFieldTooltip unknownHelp;
    Sakura::NodeField unknownFrame;
    Sakura::NodeLabel unknownText;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BASE_HH
