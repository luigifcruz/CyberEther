#include <jetstream/detail/block_interface_impl.hh>

#include <cmath>
#include <set>

namespace Jetstream {

Result Block::Interface::ValidateFormat(const Parser::Map& format, const bool metric) {
    const auto invalid = [](const std::string& reason) {
        JST_ERROR("[INTERFACE] Invalid format: {}", reason);
        return Result::ERROR;
    };
    std::string type;
    if (!format.contains("type") || Parser::Deserialize(format, "type", type) != Result::SUCCESS || type.empty()) {
        return invalid("a nonempty 'type' is required.");
    }
    std::string visibility = "public";
    if (!metric && format.contains("visibility")) {
        return invalid("'visibility' is only supported on metric descriptors.");
    }
    if (format.contains("visibility") &&
        Parser::Deserialize(format, "visibility", visibility) != Result::SUCCESS) {
        return invalid("'visibility' must be a string.");
    }
    if (visibility != "public" && visibility != "internal") {
        return invalid("'visibility' must be public or internal.");
    }
    static const std::set<std::string> controls{
        "bool", "int", "uint", "float", "range", "dropdown", "text", "multiline",
        "vector", "vector-inline", "filepicker", "filesave", "tensor-config", "markdown", "python",
    };
    if ((!metric && !controls.contains(type)) ||
        (metric && visibility != "internal" && type != "label" && type != "progressbar" && type != "table")) {
        return invalid("unsupported type '" + type + "'.");
    }
    for (const auto* key : {"unit", "step_config", "source"}) {
        std::string value;
        if (format.contains(key) && Parser::Deserialize(format, key, value) != Result::SUCCESS) {
            return invalid(std::string("'") + key + "' must be a string.");
        }
    }
    F32 scale = 1.0f;
    if (format.contains("scale") &&
        (Parser::Deserialize(format, "scale", scale) != Result::SUCCESS || !std::isfinite(scale) || scale <= 0.0f)) {
        return invalid("'scale' must be finite and positive.");
    }
    I32 precision = 2;
    if (format.contains("precision") &&
        (Parser::Deserialize(format, "precision", precision) != Result::SUCCESS || precision < 0 || precision > 16)) {
        return invalid("'precision' must be an integer between 0 and 16.");
    }
    std::string valueType = "float";
    if (format.contains("value_type")) {
        JST_CHECK(Parser::Deserialize(format, "value_type", valueType));
        if (valueType != "float" && valueType != "uint") {
            return invalid("'value_type' must be float or uint.");
        }
    }
    if (type == "range") {
        F32 minimum = 0.0f;
        F32 maximum = valueType == "uint" ? 100.0f : 1.0f;
        if ((format.contains("min") && Parser::Deserialize(format, "min", minimum) != Result::SUCCESS) ||
            (format.contains("max") && Parser::Deserialize(format, "max", maximum) != Result::SUCCESS) ||
            !std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum ||
            (valueType == "uint" && (minimum < 0.0f || std::trunc(minimum) != minimum || std::trunc(maximum) != maximum))) {
            return invalid("invalid range bounds.");
        }
    }
    if (type == "tensor-config") {
        U64 index = 0;
        if (format.contains("index") &&
            (Parser::Deserialize(format, "index", index) != Result::SUCCESS || index == std::numeric_limits<U64>::max())) {
            return invalid("invalid tensor index.");
        }
        if (Parser::Get<std::string>(format, "source").empty()) {
            return invalid("a tensor control requires a 'source' config key.");
        }
    }
    if (format.contains("collapsible")) {
        bool value;
        if (Parser::Deserialize(format, "collapsible", value) != Result::SUCCESS) {
            return invalid("'collapsible' must be a boolean.");
        }
    }
    if (format.contains("extensions")) {
        std::vector<std::string> extensions;
        if (Parser::Deserialize(format, "extensions", extensions) != Result::SUCCESS) {
            return invalid("'extensions' must be a sequence of strings.");
        }
    }
    if (type == "dropdown") {
        std::vector<Parser::Map> options;
        if (!format.contains("options") || Parser::Deserialize(format, "options", options) != Result::SUCCESS) {
            return invalid("a dropdown requires an 'options' sequence.");
        }
        for (const auto& option : options) {
            std::string label;
            std::string value;
            if (!option.contains("label") || Parser::Deserialize(option, "label", label) != Result::SUCCESS ||
                !option.contains("value") || Parser::Deserialize(option, "value", value) != Result::SUCCESS) {
                return invalid("each option requires a string label and string value.");
            }
        }
    }
    for (const auto& [key, value] : format) {
        if (!Parser::Equal(value, value)) {
            return invalid("property '" + key + "' cannot be compared reliably for change detection.");
        }
    }
    return Result::SUCCESS;
}

Block::Interface::Interface() {
    impl = std::make_shared<Impl>();
}

Block::Interface::~Interface() {
    impl.reset();
}

const Block::Interface::EntryList& Block::Interface::configs() const {
    return impl->configs;
}

const Block::Interface::EntryList& Block::Interface::inputs() const {
    return impl->inputs;
}

const Block::Interface::EntryList& Block::Interface::outputs() const {
    return impl->outputs;
}

const Block::Interface::EntryList& Block::Interface::metrics() const {
    return impl->metrics;
}

}  // namespace Jetstream
