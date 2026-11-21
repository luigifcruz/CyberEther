#include "base.hh"

namespace Jetstream {

FlowgraphNodeHeightSpec FlowgraphConfigFieldInstance::heightSpec() const {
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

void FlowgraphConfigFieldInstance::update(FlowgraphConfigFieldConfig config) {
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

bool FlowgraphConfigFieldInstance::isSimple() const {
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

Sakura::Component& FlowgraphConfigFieldInstance::component() {
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

FlowgraphConfigFieldInstance::Kind FlowgraphConfigFieldInstance::ParseKind(const std::string& value) {
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

void FlowgraphConfigFieldInstance::reset() {
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

}  // namespace Jetstream
