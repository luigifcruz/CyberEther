#include "vector.hh"

namespace Jetstream {

Result FlowgraphConfigVectorField::update(Config config) {
    if (this->config.id != config.id) {
        pending.reset();
    }
    this->config = std::move(config);
    if (this->config.format != parsedFormat) {
        parseFormat();
    }
    bool settled = false;
    if (pending && pending->complete) {
        pending.reset();
        settled = true;
    }
    const std::any nextValue = this->config.values.contains(this->config.name)
        ? this->config.values.at(this->config.name) : std::any{};
    if (settled || !Parser::Equal(nextValue, parsedValue)) {
        parsedValue.reset();
        floatValues.clear();
        uintValues.clear();
        stringValues.clear();
        if (valueType == "float") {
            JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, floatValues));
        } else if (valueType == "uint") {
            JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, uintValues));
        } else if (valueType == "string") {
            JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, stringValues));
        }
        parsedValue = nextValue;
    }
    if (pending) {
        floatValues = pending->floatValues;
        uintValues = pending->uintValues;
        stringValues = pending->stringValues;
    }
    updateChildren();
    return Result::SUCCESS;
}

Sakura::Component& FlowgraphConfigVectorField::component() {
    return grid;
}

template<typename T>
void FlowgraphConfigVectorField::submit(Parser::Map patch, std::vector<T> values) {
    if (!config.onApply) {
        return;
    }
    auto submission = std::make_shared<Pending>();
    submission->floatValues = floatValues;
    submission->uintValues = uintValues;
    submission->stringValues = stringValues;
    if constexpr (std::is_same_v<T, F32>) {
        submission->floatValues = std::move(values);
    } else if constexpr (std::is_same_v<T, U64>) {
        submission->uintValues = std::move(values);
    } else {
        submission->stringValues = std::move(values);
    }
    pending = submission;
    config.onApply(std::move(patch), false, [submission](Result) { submission->complete = true; });
}

void FlowgraphConfigVectorField::parseFormat() {
    parsedFormat = config.format;
    valueType = Parser::Get<std::string>(config.format, "value_type", "float");
    unit = Parser::Get<std::string>(config.format, "unit");
    precision = Parser::Get<I32>(config.format, "precision", 2);
    multiplier = Parser::Get<F32>(config.format, "scale", 1.0f);
    itemLabel = Parser::Get<std::string>(config.format, "item_label");
    optionLabels.clear();
    optionValues.clear();
    for (const auto& option : Parser::Get<std::vector<Parser::Map>>(config.format, "options")) {
        optionLabels.push_back(Parser::Get<std::string>(option, "label"));
        optionValues.push_back(Parser::Get<std::string>(option, "value"));
    }
    parsedValue.reset();
    floatValues.clear();
    uintValues.clear();
    stringValues.clear();
}

std::string FlowgraphConfigVectorField::entryLabel(const U64 index) const {
    return itemLabel.empty() ? config.label : jst::fmt::format("{} {}", itemLabel, index);
}

void FlowgraphConfigVectorField::updateChildren() {
    const U64 count = valueType == "float"  ? floatValues.size()
                    : valueType == "uint"   ? uintValues.size()
                    : valueType == "string" ? stringValues.size()
                                            : 0;
    const bool typeChanged = builtType != valueType;
    if (typeChanged || frames.size() != count) {
        frames.clear();
        floatInputs.clear();
        uintInputs.clear();
        stringCombos.clear();
        for (U64 i = 0; i < count; ++i) {
            frames.push_back(std::make_unique<Sakura::NodeField>());
            if (valueType == "float") {
                floatInputs.push_back(std::make_unique<Sakura::NodeFloatInput>());
            } else if (valueType == "uint") {
                uintInputs.push_back(std::make_unique<Sakura::NodeUIntInput>());
            } else {
                stringCombos.push_back(std::make_unique<Sakura::NodeCombo>());
            }
        }
        builtType = valueType;
    }

    help.update(config.id, config.help);
    std::vector<Sakura::NodeFieldGrid::Item> items;
    for (U64 i = 0; i < count; ++i) {
        Sakura::Component* content = nullptr;
        if (valueType == "float") {
            content = floatInputs[i].get();
        } else if (valueType == "uint") {
            content = uintInputs[i].get();
        } else {
            content = stringCombos[i].get();
        }
        frames[i]->update({
            .id = jst::fmt::format("{}Frame{}", config.id, i),
            .label = entryLabel(i),
            .tooltip = help.overlay(),
            .content = content,
        });
        items.push_back({
            .child = frames[i].get(),
            .fullWidth = true,
        });
    }

    if (valueType == "float") {
        for (U64 i = 0; i < count; ++i) {
            floatInputs[i]->update({
                .id = jst::fmt::format("{}Float{}", config.id, i),
                .value = floatValues[i] / multiplier,
                .unit = unit,
                .precision = precision,
                .onChange = [this, i](F32 value) {
                    Parser::Map patch;
                    auto nextValues = floatValues;
                    if (i < nextValues.size()) {
                        nextValues[i] = value * multiplier;
                    }
                    patch[config.name] = nextValues;
                    submit(std::move(patch), std::move(nextValues));
                },
            });
        }
    } else if (valueType == "uint") {
        for (U64 i = 0; i < count; ++i) {
            uintInputs[i]->update({
                .id = jst::fmt::format("{}UInt{}", config.id, i),
                .value = uintValues[i],
                .unit = unit,
                .onChange = [this, i](U64 value) {
                    Parser::Map patch;
                    auto nextValues = uintValues;
                    if (i < nextValues.size()) {
                        nextValues[i] = value;
                    }
                    patch[config.name] = nextValues;
                    submit(std::move(patch), std::move(nextValues));
                },
            });
        }
    } else if (valueType == "string") {
        for (U64 i = 0; i < count; ++i) {
            std::optional<U64> selectedIndex;
            for (U64 option = 0; option < optionValues.size(); ++option) {
                if (optionValues[option] == stringValues[i]) {
                    selectedIndex = option;
                    break;
                }
            }
            stringCombos[i]->update({
                .id = jst::fmt::format("{}Combo{}", config.id, i),
                .options = optionLabels,
                .value = selectedIndex ? optionLabels[*selectedIndex] : std::string("Selection unavailable"),
                .selectedIndex = selectedIndex,
                .onSelect = [this, i](U64 index) {
                    Parser::Map patch;
                    auto nextValues = stringValues;
                    if (i < nextValues.size() && index < optionValues.size()) {
                        nextValues[i] = optionValues[index];
                    }
                    patch[config.name] = nextValues;
                    submit(std::move(patch), std::move(nextValues));
                },
            });
        }
    }

    grid.update({
        .id = config.id + "Grid",
        .items = std::move(items),
    });
}

}  // namespace Jetstream
