#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_INLINE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_INLINE_HH

#include "types.hh"

#include <cctype>

namespace Jetstream {

struct FlowgraphConfigVectorInlineField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            parseFormat();
        }
        const std::any nextValue = this->config.values.contains(this->config.name)
            ? this->config.values.at(this->config.name) : std::any{};
        if (!Parser::Equal(nextValue, parsedValue)) {
            JST_CHECK(formatCurrentValue());
            parsedValue = nextValue;
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
        return Result::SUCCESS;
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            input.render(ctx);
        });
    }

 private:
    void parseFormat() {
        parsedFormat = config.format;
        valueType = Parser::Get<std::string>(config.format, "value_type", "float");
        unit = Parser::Get<std::string>(config.format, "unit");
        precision = Parser::Get<I32>(config.format, "precision", 2);
        multiplier = Parser::Get<F32>(config.format, "scale", 1.0f);
        parsedValue.reset();
        buffer = "[]";
    }

    Result formatCurrentValue() {
        if (valueType == "float") {
            std::vector<F32> values;
            JST_CHECK(Parser::Deserialize(config.values, config.name, values));
            std::vector<std::string> formattedValues;
            formattedValues.reserve(values.size());
            for (const auto value : values) {
                formattedValues.push_back(jst::fmt::format("{:.{}f}", value / multiplier, precision));
            }
            buffer = jst::fmt::format("[{}]", jst::fmt::join(formattedValues, ", "));
        } else if (valueType == "uint") {
            std::vector<U64> values;
            JST_CHECK(Parser::Deserialize(config.values, config.name, values));
            buffer = jst::fmt::format("[{}]", jst::fmt::join(values, ", "));
        }
        return Result::SUCCESS;
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

        Parser::Map patch;
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
                for (auto& value : parsedValues) {
                    value *= multiplier;
                }
                patch[config.name] = parsedValues;
                changed = true;
            }
        } else if (valueType == "uint") {
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
                patch[config.name] = parsedValues;
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
            config.onApply(std::move(patch), false);
        }
        return true;
    }

    Config config;
    Parser::Map parsedFormat;
    std::any parsedValue;
    F32 multiplier = 1.0f;
    std::string valueType = "float";
    std::string unit;
    int precision = 2;
    std::string buffer = "[]";
    Sakura::NodeField frame;
    Sakura::NodeTextInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_INLINE_HH
