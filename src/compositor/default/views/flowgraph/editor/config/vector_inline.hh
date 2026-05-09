#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_INLINE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_INLINE_HH

#include "types.hh"

#include <cctype>

// TODO: Cleanup parsing.

namespace Jetstream {

struct FlowgraphConfigVectorInlineField : public Sakura::Component {
    using Config = FlowgraphConfigFieldConfig;

    void update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            parseFormat();
        }
        if (this->config.encoded != parsedEncoded) {
            buffer = formatCurrentValue();
            parsedEncoded = this->config.encoded;
        }
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
        });
        input.update({
            .id = this->config.id + "Input",
            .value = buffer,
            .unit = unit,
            .submit = Sakura::TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                applyBuffer(value);
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            input.render(ctx);
        });
    }

 private:
    void parseFormat() {
        parsedFormat = config.format;
        const auto parts = Parser::SplitString(config.format, ":");
        valueType = (parts.size() > 1) ? parts[1] : "float";
        unit = (parts.size() > 2) ? parts[2] : "";
        precision = (parts.size() > 3 && !parts[3].empty()) ? std::stoi(parts[3]) : 2;
    }

    std::string formatCurrentValue() const {
        if (config.encoded.empty()) {
            return "[]";
        }

        try {
            if (valueType == "float") {
                std::vector<F32> values;
                if (Parser::StringToTyped(config.encoded, values) == Result::SUCCESS) {
                    const F32 multiplier = ConfigUnitMultiplier(unit);
                    std::vector<std::string> formattedValues;
                    formattedValues.reserve(values.size());
                    for (const auto value : values) {
                        formattedValues.push_back(jst::fmt::format("{:.{}f}", value / multiplier, precision));
                    }
                    return jst::fmt::format("[{}]", jst::fmt::join(formattedValues, ", "));
                }
            } else if (valueType == "int") {
                std::vector<U64> values;
                if (Parser::StringToTyped(config.encoded, values) == Result::SUCCESS) {
                    return jst::fmt::format("[{}]", jst::fmt::join(values, ", "));
                }
            }
        } catch (...) {
        }

        return config.encoded.empty() ? "[]" : config.encoded;
    }

    static bool normalizeVectorInput(const std::string& text, std::string& normalized, std::string& error) {
        const auto trim = [](std::string s) {
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
            return s;
        };

        std::string trimmed = trim(text);
        if (trimmed.empty()) {
            normalized = "[]";
            return true;
        }

        if (trimmed.front() == '[') {
            if (trimmed.back() != ']') {
                error = "Vector input must end with ']'.";
                return false;
            }
            trimmed = trim(trimmed.substr(1, trimmed.size() - 2));
        } else if (trimmed.back() == ']') {
            error = "Vector input must start with '['.";
            return false;
        }

        std::vector<std::string> entries;
        std::stringstream ss(trimmed);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trim(token);
            if (token.empty()) {
                error = "Vector entries cannot be empty.";
                return false;
            }
            entries.push_back(token);
        }

        normalized = jst::fmt::format("[{}]", jst::fmt::join(entries, ", "));
        return true;
    }

    bool applyBuffer(const std::string& nextBuffer) {
        buffer = nextBuffer;
        std::string normalizedBuffer;
        std::string error;
        normalizeVectorInput(buffer, normalizedBuffer, error);

        auto values = config.values;
        bool changed = false;
        if (valueType == "float") {
            std::vector<F32> parsedValues;
            if (error.empty()) {
                try {
                    if (normalizedBuffer != "[]" && Parser::StringToTyped(normalizedBuffer, parsedValues) != Result::SUCCESS) {
                        error = "Invalid float value in vector.";
                    }
                } catch (...) {
                    error = "Invalid float value in vector.";
                }
            }

            if (error.empty()) {
                const F32 multiplier = ConfigUnitMultiplier(unit);
                for (auto& value : parsedValues) {
                    value *= multiplier;
                }
                values[config.name] = parsedValues;
                changed = true;
            }
        } else if (valueType == "int") {
            std::vector<U64> parsedValues;
            if (error.empty()) {
                try {
                    if (normalizedBuffer != "[]" && Parser::StringToTyped(normalizedBuffer, parsedValues) != Result::SUCCESS) {
                        error = "Invalid integer value in vector.";
                    }
                } catch (...) {
                    error = "Invalid integer value in vector.";
                }
            }

            if (error.empty()) {
                values[config.name] = parsedValues;
                changed = true;
            }
        } else {
            error = jst::fmt::format("Unknown vector-inline field type '{}'.", valueType);
        }

        if (!changed) {
            if (config.onError) {
                config.onError(Result::ERROR,
                               jst::fmt::format("{}: {}",
                                                config.label,
                                                error.empty() ? "Use a vector like [x, y, z]." : error));
            }
            return false;
        }

        if (config.onApply) {
            config.onApply(std::move(values), false);
        }
        return true;
    }

    Config config;
    std::string parsedFormat;
    std::string parsedEncoded;
    std::string valueType = "float";
    std::string unit;
    int precision = 2;
    std::string buffer = "[]";
    Sakura::NodeField frame;
    Sakura::NodeTextInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_INLINE_HH
