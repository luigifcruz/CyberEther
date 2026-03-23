#include "imgui.h"
#include "jetstream/compositor.hh"
#include "jetstream/detail/compositor_impl.hh"

#include "jetstream/render/tools/imgui_icons_ext.hh"
#include "jetstream/render/tools/imnodes.h"
#include "jetstream/render/tools/imgui_markdown.hh"
#include "jetstream/render/tools/imgui_notify_ext.h"
#include "jetstream/render/tools/imgui_code_editor.hh"

#include "jetstream/platform.hh"
#include "jetstream/parser.hh"
#include "jetstream/registry.hh"
#include "jetstream/benchmark.hh"
#include "jetstream/types.hh"
#include "jetstream/block_interface.hh"
#include "jetstream/module_surface.hh"
#include "jetstream/instance_remote.hh"

#include <qrencode.h>

#include "resources/fonts/compressed_jbmm.hh"
#include "resources/fonts/compressed_jbmb.hh"
#include "resources/fonts/compressed_fa.hh"
#include "resources/flowgraphs/base.hh"

#include <algorithm>
#include <any>
#include <cctype>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <sstream>
#include <optional>
#include <regex>
#include <set>
#include <vector>
#include <unordered_map>
#include <variant>

namespace Jetstream {

template<class... Ts>
struct Overloaded : Ts... { using Ts::operator()...; };
template<class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

//
// Field Rendering Helpers
//

static inline std::string TrimFieldString(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

static inline std::string CleanUserMessage(std::string message) {
    static const std::regex prefix(R"(^\[[^\]]+\]\s*)");
    return std::regex_replace(message, prefix, "");
}

static inline void NotifyResultClean(const Result value, const std::string& message = "") {
    const auto resolveMessage = [&](const std::string& fallback) {
        return CleanUserMessage(message.empty() ? fallback : message);
    };

    if (value == Result::ERROR) {
        const auto text = resolveMessage(JST_LOG_LAST_ERROR());
        ImGui::InsertNotification({ ImGuiToastType_Error, 5000, text.c_str() });
    } else if (value == Result::FATAL) {
        const auto text = resolveMessage(JST_LOG_LAST_FATAL());
        ImGui::InsertNotification({ ImGuiToastType_Error, 5000, text.c_str() });
    } else if (value == Result::WARNING) {
        const auto text = resolveMessage(JST_LOG_LAST_WARNING());
        ImGui::InsertNotification({ ImGuiToastType_Warning, 5000, text.c_str() });
    } else if (value == Result::SUCCESS) {
        ImGui::InsertNotification({ ImGuiToastType_Success, 1000, "" });
    } else if (value == Result::INCOMPLETE) {
        const auto text = resolveMessage(JST_LOG_LAST_ERROR());
        ImGui::InsertNotification({ ImGuiToastType_Info, 1000, text.c_str() });
    }
}

struct FieldContext {
    Parser::Map& cfg;
    F32 scalingFactor;
    const std::unordered_map<std::string, ImVec4>& colorMap;
    std::string label;
    std::string help;
    bool silent = false;
    std::function<void(std::string, std::string)> asyncApply;

    void renderHeader(std::optional<std::string> index = std::nullopt) {
        const auto& bgColor = ImGui::ColorConvertFloat4ToU32(colorMap.at("card"));
        const auto& unitColor = ImGui::ColorConvertFloat4ToU32(colorMap.at("text_secondary"));

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 4.0f * scalingFactor);

        const ImVec2 pos = ImGui::GetCursorScreenPos();
        const F32 height = ImGui::GetTextLineHeight() + 12.0f * scalingFactor;
        const ImVec2 min = ImVec2(pos.x, pos.y - 2.0f * scalingFactor);
        const ImVec2 max = ImVec2(pos.x + ImGui::GetContentRegionAvail().x, pos.y + height);

        ImGui::GetWindowDrawList()->AddRectFilled(min, max, bgColor, ImGui::GetStyle().FrameRounding);

        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6.0f * scalingFactor);
        ImGui::PushFont(ImGui::GetFont(), ImGui::GetFontSize() * 0.75f);
        ImGui::PushStyleColor(ImGuiCol_Text, unitColor);
        ImGui::TextUnformatted(jst::fmt::format("{} {}", label, index.value_or("")).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        if (!help.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
            ImGui::BeginTooltip();
            ImGui::TextUnformatted(help.c_str());
            ImGui::EndTooltip();
        }

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 8.0f * scalingFactor);
    }

    void renderUnitSuffix(const std::string& unitStr) const {
        if (unitStr.empty()) return;
        const auto unitColor = ImGui::ColorConvertFloat4ToU32(colorMap.at("text_secondary"));
        const ImVec2 inputPos = ImGui::GetItemRectMin();
        const ImVec2 inputSize = ImGui::GetItemRectSize();
        const ImVec2 textSize = ImGui::CalcTextSize(unitStr.c_str());
        const ImVec2 textPos = ImVec2(inputPos.x + inputSize.x - textSize.x - 3.0f * scalingFactor,
                                      inputPos.y + (inputSize.y - textSize.y) * 0.5f);
        ImGui::GetWindowDrawList()->AddText(textPos, unitColor, unitStr.c_str());
    }

    bool renderFloatValue(F32& value, const std::string& unit = "", int precision = 2) {
        const F32 multiplier = getUnitMultiplier(unit);
        F32 displayValue = value / multiplier;

        char fmt[16];
        snprintf(fmt, sizeof(fmt), "%%.%df", precision);

        ImGui::SetNextItemWidth(-1);

        bool changed = false;
        if (ImGui::InputFloat("##float", &displayValue, 0.0f, 0.0f, fmt, ImGuiInputTextFlags_EnterReturnsTrue)) {
            value = displayValue * multiplier;
            changed = true;
        }

        renderUnitSuffix(unit);

        return changed;
    }

    bool renderIntValue(U64& value, const std::string& unit = "") {
        ImGui::SetNextItemWidth(-1);

        bool changed = false;
        if (ImGui::InputScalar("##int", ImGuiDataType_U64, &value, nullptr, nullptr, nullptr, ImGuiInputTextFlags_EnterReturnsTrue)) {
            changed = true;
        }

        renderUnitSuffix(unit);

        return changed;
    }

    static F32 getUnitMultiplier(const std::string& unit) {
        if (unit == "GHz") return 1e9f;
        if (unit == "MHz") return 1e6f;
        if (unit == "kHz") return 1e3f;
        return 1.0f;
    }
};

// Format: `dropdown:key1(Label1),key2(Label2),...`
static inline bool RenderFieldDropdown(const std::string& name,
                                       const std::vector<std::string>& parts,
                                       const std::string& currentValue,
                                       FieldContext& ctx) {
    ctx.renderHeader();

    std::vector<std::string> keys;
    std::vector<std::string> labels;

    const std::string options = (parts.size() > 1) ? parts[1] : "";
    for (auto& token : Parser::SplitString(options, ",")) {
        token = TrimFieldString(token);
        if (token.empty()) continue;

        const auto open = token.find('(');
        const auto close = token.rfind(')');
        if (open != std::string::npos && close != std::string::npos && close > open) {
            keys.push_back(TrimFieldString(token.substr(0, open)));
            labels.push_back(TrimFieldString(token.substr(open + 1, close - open - 1)));
        } else {
            keys.push_back(token);
            labels.push_back(token);
        }
    }

    std::vector<const char*> labelPtrs;
    labelPtrs.reserve(labels.size());
    for (const auto& l : labels) {
        labelPtrs.push_back(l.c_str());
    }

    int currentIdx = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (keys[i] == currentValue) {
            currentIdx = static_cast<int>(i);
            break;
        }
    }

    const auto bgColor = ImGui::ColorConvertFloat4ToU32(ctx.colorMap.at("card"));

    ImGui::PushID(name.c_str());
    ImGui::SetNextItemWidth(-1);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, bgColor);
    ImGui::PushStyleColor(ImGuiCol_Button, bgColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bgColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, bgColor);

    bool changed = false;
    if (keys.empty()) {
        ImGui::BeginDisabled();
        int dummy = 0;
        const char* emptyLabel = "No options";
        ImGui::Combo("##dropdown", &dummy, &emptyLabel, 1);
        ImGui::EndDisabled();
    } else if (ImGui::Combo("##dropdown", &currentIdx, labelPtrs.data(), static_cast<int>(labelPtrs.size()))) {
        ctx.cfg[name] = keys[currentIdx];
        changed = true;
    }

    ImGui::PopStyleColor(4);
    ImGui::PopID();

    return changed;
}

// Format: `float:unit:precision`
static inline bool RenderFieldFloat(const std::string& name,
                                    const std::vector<std::string>& parts,
                                    const std::string& encoded,
                                    FieldContext& ctx) {
    ctx.renderHeader();

    F32 value = 0.0f;
    if (!encoded.empty()) {
        Parser::StringToTyped(encoded, value);
    }

    const std::string unit = (parts.size() > 1) ? parts[1] : "";
    const int precision = (parts.size() > 2 && !parts[2].empty()) ? std::stoi(parts[2]) : 2;

    ImGui::PushID(name.c_str());

    bool changed = false;
    if (ctx.renderFloatValue(value, unit, precision)) {
        ctx.cfg[name] = value;
        changed = true;
    }

    ImGui::PopID();

    return changed;
}

// Format: `int:unit`
static inline bool RenderFieldInt(const std::string& name,
                                  const std::vector<std::string>& parts,
                                  const std::string& encoded,
                                  FieldContext& ctx) {
    ctx.renderHeader();

    U64 value = 0;
    if (!encoded.empty()) {
        Parser::StringToTyped(encoded, value);
    }

    const std::string unit = (parts.size() > 1) ? parts[1] : "";

    ImGui::PushID(name.c_str());

    bool changed = false;
    if (ctx.renderIntValue(value, unit)) {
        ctx.cfg[name] = value;
        changed = true;
    }

    ImGui::PopID();

    return changed;
}

// Format: `vector:float:unit:precision` or `vector:int:unit`
static inline bool RenderFieldVector(const std::string& name,
                                     const std::vector<std::string>& parts,
                                     const std::string& encoded,
                                     FieldContext& ctx) {
    const std::string valueType = (parts.size() > 1) ? parts[1] : "float";

    ImGui::PushID(name.c_str());

    bool changed = false;

    if (valueType == "float") {
        std::vector<F32> values;
        if (!encoded.empty()) {
            Parser::StringToTyped(encoded, values);
        }

        if (values.empty()) {
            ImGui::PopID();
            return false;
        }

        const std::string unit = (parts.size() > 2) ? parts[2] : "";
        const int precision = (parts.size() > 3 && !parts[3].empty()) ? std::stoi(parts[3]) : 2;

        for (U64 i = 0; i < values.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));

            ctx.renderHeader(std::to_string(i));

            if (ctx.renderFloatValue(values[i], unit, precision)) {
                ctx.cfg[name] = values;
                changed = true;
            }

            ImGui::PopID();
        }
    } else if (valueType == "int") {
        std::vector<U64> values;
        if (!encoded.empty()) {
            Parser::StringToTyped(encoded, values);
        }

        if (values.empty()) {
            ImGui::PopID();
            return false;
        }

        const std::string unit = (parts.size() > 2) ? parts[2] : "";

        for (U64 i = 0; i < values.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));

            ctx.renderHeader(std::to_string(i));

            if (ctx.renderIntValue(values[i], unit)) {
                ctx.cfg[name] = values;
                changed = true;
            }

            ImGui::PopID();
        }
    } else {
        JST_ERROR("[COMPOSITOR_IMPL_DEFAULT] Unknown vector field type '{}' for config '{}'", valueType, name);
    }

    ImGui::PopID();

    return changed;
}

// Format: `text`
static inline bool RenderFieldText(const std::string& name,
                                   const std::vector<std::string>&,
                                   const std::string& currentValue,
                                   FieldContext& ctx) {
    ctx.renderHeader();

    std::string value = currentValue;

    ImGui::PushID(name.c_str());
    ImGui::SetNextItemWidth(-1);

    bool changed = false;
    if (ImGui::InputText("##text", &value, ImGuiInputTextFlags_EnterReturnsTrue)) {
        ctx.cfg[name] = value;
        changed = true;
    }
    ImGui::PopID();

    return changed;
}

// Format: `multiline:collapsible`
static inline bool RenderFieldMultiline(const std::string& name,
                                        const std::vector<std::string>& parts,
                                        const std::string& currentValue,
                                        FieldContext& ctx) {
    ctx.renderHeader();

    const auto bgColor = ImGui::ColorConvertFloat4ToU32(ctx.colorMap.at("card"));
    const bool collapsible = parts.size() > 1 && parts[1] == "collapsible";

    const ImGuiID blockEditingId = ImGui::GetID("##multiline_editing");

    ImGui::PushID(name.c_str());
    bool changed = false;

    if (collapsible) {
        const ImGuiID editingId = ImGui::GetID("##editing");
        ImGuiStorage* storage = ImGui::GetStateStorage();
        bool editing = storage->GetBool(editingId, false);

        storage->SetBool(blockEditingId, editing);

        static std::unordered_map<ImGuiID, std::string> buffers;

        if (editing) {
            auto it = buffers.find(editingId);
            if (it == buffers.end()) {
                it = buffers.emplace(editingId, currentValue).first;
            }

            ImGui::InputTextCodeEditor("##multiline", &it->second,
                                       ImVec2(0, 200));

            ImGui::SetCursorPosY(ImGui::GetCursorPosY()
                                 - ImGui::GetStyle().ItemSpacing.y * 0.5f);
            ImGui::PushStyleColor(ImGuiCol_Button, bgColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bgColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, bgColor);
            ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign,
                                ImVec2(0.5f, 0.5f));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                                ImVec2(ImGui::GetStyle().FramePadding.x,
                                       4.0f));

            if (ImGui::Button("Done", ImVec2(-1, 0))) {
                ctx.cfg[name] = it->second;
                changed = true;
                buffers.erase(it);
                storage->SetBool(editingId, false);
            }

            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(3);
        } else {
            buffers.erase(editingId);

            const F32 pad = ImGui::GetStyle().FramePadding.x;
            const F32 iconScale = 0.6f;
            const F32 iconFontSize = ImGui::GetFontSize() * iconScale;
            const F32 iconWidth = ImGui::GetFont()->CalcTextSizeA(iconFontSize,
                                                                  FLT_MAX,
                                                                  0.0f,
                                                                  ICON_FA_PEN_TO_SQUARE).x;

            ImGui::SetNextItemWidth(-1);
            ImGui::PushStyleColor(ImGuiCol_FrameBg, bgColor);
            ImGui::PushStyleColor(ImGuiCol_Button, bgColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bgColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, bgColor);
            ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));

            if (ImGui::Button("Edit", ImVec2(-1, 0))) {
                storage->SetBool(editingId, true);
            }

            const ImVec2 rectMin = ImGui::GetItemRectMin();
            const ImVec2 rectMax = ImGui::GetItemRectMax();
            const F32 frameHeight = rectMax.y - rectMin.y;
            const F32 iconX = rectMax.x - pad * 0.4f - iconWidth;
            const F32 iconY = rectMin.y + (frameHeight - iconFontSize) * 0.5f;

            const ImVec2 bgMin(iconX - pad * 0.5f, rectMin.y);
            ImGui::GetWindowDrawList()->AddRectFilled(bgMin, rectMax, bgColor);
            ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(),
                                                iconFontSize,
                                                ImVec2(iconX, iconY),
                                                ImGui::GetColorU32(ImGuiCol_TextDisabled),
                                                ICON_FA_PEN_TO_SQUARE);

            ImGui::PopStyleVar();
            ImGui::PopStyleColor(4);
        }
    } else {
        std::string value = currentValue;
        if (ImGui::InputTextCodeEditor("##multiline", &value,
                                       ImVec2(0, 200))) {
            ctx.cfg[name] = value;
            changed = true;
        }
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            ctx.cfg[name] = value;
            changed = true;
        }
    }

    ImGui::PopID();
    return changed;
}

// Format: `range:min:max:unit:type`
static inline bool RenderFieldRange(const std::string& name,
                                    const std::vector<std::string>& parts,
                                    const std::string& currentValue,
                                    FieldContext& ctx) {
    ctx.renderHeader();

    const auto bgColor = ImGui::ColorConvertFloat4ToU32(ctx.colorMap.at("card"));
    const auto unitColor = ImGui::ColorConvertFloat4ToU32(ctx.colorMap.at("text_secondary"));
    const std::string unit = (parts.size() > 3) ? parts[3] : "";
    const std::string type = (parts.size() > 4) ? parts[4] : "";
    const bool isInt = (type == "int");

    const F32 pad = ImGui::GetStyle().FramePadding.x;
    const F32 rounding = ImGui::GetStyle().FrameRounding;
    char valueText[64];

    F32 minVal, maxVal, value, fraction;

    if (isInt) {
        const U64 minValInt = (parts.size() > 1 && !parts[1].empty()) ? std::stoull(parts[1]) : 0;
        const U64 maxValInt = (parts.size() > 2 && !parts[2].empty()) ? std::stoull(parts[2]) : 100;
        minVal = static_cast<F32>(minValInt);
        maxVal = static_cast<F32>(maxValInt);

        U64 valueInt = minValInt;
        if (!currentValue.empty()) {
            Parser::StringToTyped(currentValue, valueInt);
        }
        value = static_cast<F32>(valueInt);
        snprintf(valueText, sizeof(valueText), "%llu", static_cast<unsigned long long>(valueInt));
    } else {
        minVal = (parts.size() > 1 && !parts[1].empty()) ? std::stof(parts[1]) : 0.0f;
        maxVal = (parts.size() > 2 && !parts[2].empty()) ? std::stof(parts[2]) : 1.0f;

        value = 0.0f;
        if (!currentValue.empty()) {
            Parser::StringToTyped(currentValue, value);
        }
        snprintf(valueText, sizeof(valueText), "%.0f", value);
    }

    ImGui::PushID(name.c_str());

    const F32 availWidth = ImGui::GetContentRegionAvail().x;
    const F32 frameHeight = ImGui::GetFrameHeight();

    ImGui::InvisibleButton("##range", ImVec2(availWidth, frameHeight));

    const ImVec2 rectMin = ImGui::GetItemRectMin();
    const ImVec2 rectMax = ImGui::GetItemRectMax();

    ImGui::GetWindowDrawList()->AddRectFilled(rectMin, rectMax, bgColor, rounding);

    const F32 range = maxVal - minVal;
    fraction = (range != 0.0f) ? (value - minVal) / range : 0.0f;
    fraction = std::clamp(fraction, 0.0f, 1.0f);

    bool changed = false;
    if (ImGui::IsItemActive()) {
        const F32 mouseX = ImGui::GetIO().MousePos.x;
        const F32 newFraction = std::clamp((mouseX - rectMin.x) / (rectMax.x - rectMin.x), 0.0f, 1.0f);
        const F32 newValue = minVal + newFraction * range;
        if (newValue != value) {
            value = newValue;
            fraction = newFraction;
            if (isInt) {
                ctx.cfg[name] = static_cast<U64>(value);
                snprintf(valueText, sizeof(valueText), "%llu", static_cast<unsigned long long>(value));
            } else {
                ctx.cfg[name] = value;
                snprintf(valueText, sizeof(valueText), "%.0f", value);
            }
            ctx.silent = true;
            changed = true;
        }
    }

    const F32 sliderWidth = (rectMax.x - rectMin.x) * fraction;
    const ImVec4 bgColorVec = ImGui::ColorConvertU32ToFloat4(bgColor);
    const ImU32 sliderColor = ImGui::GetColorU32(ImVec4(bgColorVec.x * 1.5f, bgColorVec.y * 1.5f, bgColorVec.z * 1.5f, bgColorVec.w));

    const F32 knobWidth = 8.0f * ctx.scalingFactor;
    const F32 knobX = std::clamp(rectMin.x + sliderWidth - knobWidth * 0.5f, rectMin.x, rectMax.x - knobWidth);

    ImGui::GetWindowDrawList()->AddRectFilled(rectMin, ImVec2(knobX + knobWidth, rectMax.y), sliderColor, rounding);

    const ImU32 knobColor = ImGui::GetColorU32(ImVec4(bgColorVec.x * 2.0f, bgColorVec.y * 2.0f, bgColorVec.z * 2.0f, bgColorVec.w));
    ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(knobX, rectMin.y), ImVec2(knobX + knobWidth, rectMax.y), knobColor, rounding);

    const ImVec2 textPos(rectMin.x + pad, rectMin.y + (frameHeight - ImGui::GetFontSize()) * 0.5f);
    ImGui::GetWindowDrawList()->AddText(textPos, ImGui::GetColorU32(ImGuiCol_Text), valueText);

    if (!unit.empty()) {
        const ImVec2 unitTextSize = ImGui::CalcTextSize(unit.c_str());
        const ImVec2 unitTextPos = ImVec2(rectMax.x - unitTextSize.x - 3.0f * ctx.scalingFactor,
                                          rectMin.y + (frameHeight - unitTextSize.y) * 0.5f);
        ImGui::GetWindowDrawList()->AddText(unitTextPos, unitColor, unit.c_str());
    }

    ImGui::PopID();

    return changed;
}

// Format: `bool`
static inline bool RenderFieldBool(const std::string& name,
                                   const std::vector<std::string>&,
                                   const std::string& currentValue,
                                   FieldContext& ctx) {
    ctx.renderHeader();

    const auto bgColor = ImGui::ColorConvertFloat4ToU32(ctx.colorMap.at("card"));
    bool value = (currentValue == "true" || currentValue == "1");
    const std::string displayText = value ? "Enabled" : "Disabled";
    const char* icon = value ? ICON_FA_TOGGLE_ON : ICON_FA_TOGGLE_OFF;

    const F32 pad = ImGui::GetStyle().FramePadding.x;
    const F32 iconScale = 0.8f;
    const F32 iconFontSize = ImGui::GetFontSize() * iconScale;
    const F32 iconWidth = ImGui::GetFont()->CalcTextSizeA(iconFontSize, FLT_MAX, 0.0f, icon).x;

    ImGui::PushID(name.c_str());
    ImGui::SetNextItemWidth(-1);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, bgColor);
    ImGui::PushStyleColor(ImGuiCol_Button, bgColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bgColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, bgColor);
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));

    bool changed = false;
    if (ImGui::Button(displayText.c_str(), ImVec2(-1, 0))) {
        ctx.cfg[name] = !value;
        changed = true;
    }

    const ImVec2 rectMin = ImGui::GetItemRectMin();
    const ImVec2 rectMax = ImGui::GetItemRectMax();
    const F32 frameHeight = rectMax.y - rectMin.y;
    const F32 iconX = rectMax.x - pad * 0.4f - iconWidth;
    const F32 iconY = rectMin.y + (frameHeight - iconFontSize) * 0.5f;

    const ImVec2 bgMin(iconX - pad * 0.5f, rectMin.y);
    const ImVec2 bgMax(rectMax.x, rectMax.y);
    ImGui::GetWindowDrawList()->AddRectFilled(bgMin, bgMax, bgColor);

    const ImU32 iconColor = value ? ImGui::GetColorU32(ImVec4(0.4f, 0.8f, 0.4f, 1.0f))
                                  : ImGui::GetColorU32(ImGuiCol_TextDisabled);
    ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), iconFontSize, ImVec2(iconX, iconY), iconColor, icon);

    ImGui::PopStyleVar();
    ImGui::PopStyleColor(4);
    ImGui::PopID();

    return changed;
}

// Format: `filepicker:ext1,ext2,...`
static inline bool RenderFieldFilePicker(const std::string& name,
                                         const std::vector<std::string>& parts,
                                         const std::string& currentValue,
                                         FieldContext& ctx) {
    ctx.renderHeader();

    const auto bgColor = ImGui::ColorConvertFloat4ToU32(ctx.colorMap.at("card"));
    std::vector<std::string> extensions;
    if (parts.size() > 1 && !parts[1].empty()) {
        extensions = Parser::SplitString(parts[1], ",");
    }

    std::string filename = "Select file...";
    if (!currentValue.empty()) {
        const auto pos = currentValue.find_last_of("/\\");
        filename = (pos != std::string::npos) ? currentValue.substr(pos + 1) : currentValue;
    }

    const F32 pad = ImGui::GetStyle().FramePadding.x;
    const F32 iconScale = 0.7f;
    const F32 iconFontSize = ImGui::GetFontSize() * iconScale;
    const F32 iconWidth = ImGui::GetFont()->CalcTextSizeA(iconFontSize, FLT_MAX, 0.0f, ICON_FA_FOLDER_OPEN).x;

    ImGui::PushID(name.c_str());
    ImGui::SetNextItemWidth(-1);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, bgColor);
    ImGui::PushStyleColor(ImGuiCol_Button, bgColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bgColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, bgColor);
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));

    if (ImGui::Button(filename.c_str(), ImVec2(-1, 0))) {
        std::string pickedPath;
        Platform::PickFile(pickedPath, extensions, [name, async = ctx.asyncApply](std::string path) {
            async(name, std::move(path));
        });
    }

    const ImVec2 rectMin = ImGui::GetItemRectMin();
    const ImVec2 rectMax = ImGui::GetItemRectMax();
    const F32 frameHeight = rectMax.y - rectMin.y;
    const F32 arrowPad = ImGui::GetStyle().FramePadding.y;
    const F32 iconX = rectMax.x - pad * 0.4f - iconWidth;
    const F32 iconY = rectMin.y + (frameHeight - iconFontSize) * 0.5f + arrowPad * 0.1f;

    const ImVec2 bgMin(iconX - pad * 0.5f, rectMin.y);
    const ImVec2 bgMax(rectMax.x, rectMax.y);
    ImGui::GetWindowDrawList()->AddRectFilled(bgMin, bgMax, bgColor);

    ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(),
                                        iconFontSize,
                                        ImVec2(iconX, iconY),
                                        ImGui::GetColorU32(ImGuiCol_Text),
                                        ICON_FA_FOLDER_OPEN);

    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort) && !currentValue.empty()) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(currentValue.c_str());
        ImGui::EndTooltip();
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleColor(4);
    ImGui::PopID();

    return false;
}

// Format: `filesave:ext1,ext2,...`
static inline bool RenderFieldFileSave(const std::string& name,
                                       const std::vector<std::string>&,
                                       const std::string& currentValue,
                                       FieldContext& ctx) {
    ctx.renderHeader();

    const auto bgColor = ImGui::ColorConvertFloat4ToU32(ctx.colorMap.at("card"));
    std::string filename = "Select file...";
    if (!currentValue.empty()) {
        const auto pos = currentValue.find_last_of("/\\");
        filename = (pos != std::string::npos) ? currentValue.substr(pos + 1) : currentValue;
    }

    const F32 pad = ImGui::GetStyle().FramePadding.x;
    const F32 iconScale = 0.7f;
    const F32 iconFontSize = ImGui::GetFontSize() * iconScale;
    const F32 iconWidth = ImGui::GetFont()->CalcTextSizeA(iconFontSize, FLT_MAX, 0.0f, ICON_FA_FLOPPY_DISK).x;

    ImGui::PushID(name.c_str());
    ImGui::SetNextItemWidth(-1);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, bgColor);
    ImGui::PushStyleColor(ImGuiCol_Button, bgColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bgColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, bgColor);
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));

    if (ImGui::Button(filename.c_str(), ImVec2(-1, 0))) {
        std::string pickedPath;
        Platform::SaveFile(pickedPath, [name, async = ctx.asyncApply](std::string path) {
            async(name, std::move(path));
        });
    }

    const ImVec2 rectMin = ImGui::GetItemRectMin();
    const ImVec2 rectMax = ImGui::GetItemRectMax();
    const F32 frameHeight = rectMax.y - rectMin.y;
    const F32 arrowPad = ImGui::GetStyle().FramePadding.y;
    const F32 iconX = rectMax.x - pad * 0.4f - iconWidth;
    const F32 iconY = rectMin.y + (frameHeight - iconFontSize) * 0.5f + arrowPad * 0.1f;

    const ImVec2 bgMin(iconX - pad * 0.5f, rectMin.y);
    const ImVec2 bgMax(rectMax.x, rectMax.y);
    ImGui::GetWindowDrawList()->AddRectFilled(bgMin, bgMax, bgColor);

    ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(),
                                        iconFontSize,
                                        ImVec2(iconX, iconY),
                                        ImGui::GetColorU32(ImGuiCol_Text),
                                        ICON_FA_FLOPPY_DISK);

    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort) && !currentValue.empty()) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(currentValue.c_str());
        ImGui::EndTooltip();
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleColor(4);
    ImGui::PopID();

    return false;
}

using FieldRenderer = bool(*)(const std::string&,
                              const std::vector<std::string>&,
                              const std::string&,
                              FieldContext&);

static const std::unordered_map<std::string, FieldRenderer> fieldRenderers = {
    {"dropdown",   RenderFieldDropdown},
    {"float",      RenderFieldFloat},
    {"int",        RenderFieldInt},
    {"vector",     RenderFieldVector},
    {"filepicker", RenderFieldFilePicker},
    {"filesave",   RenderFieldFileSave},
    {"bool",       RenderFieldBool},
    {"range",      RenderFieldRange},
    {"multiline",  RenderFieldMultiline},
    {"text",       RenderFieldText},
};

//
// Metric Rendering Helpers
//

struct MetricContext {
    F32 scalingFactor;
    const std::unordered_map<std::string, ImVec4>& colorMap;
    std::string label;
    std::string help;
    const ImGui::MarkdownConfig& markdownConfig;

    void renderHeader() {
        const auto& unitColor = ImGui::ColorConvertFloat4ToU32(colorMap.at("text_secondary"));

        if (!label.empty()) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6.0f * scalingFactor);
            ImGui::PushFont(ImGui::GetFont(), ImGui::GetFontSize() * 0.75f);
            ImGui::PushStyleColor(ImGuiCol_Text, unitColor);
            ImGui::TextUnformatted(label.c_str());
            ImGui::PopStyleColor();
            ImGui::PopFont();

            if (!help.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(help.c_str());
                ImGui::EndTooltip();
            }

            ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 4.0f * scalingFactor);
        }
    }
};

// Format: `progressbar`
static inline void RenderMetricProgressBar(const Block::Interface::Entry& entry,
                                           MetricContext& ctx) {
    ctx.renderHeader();
    if (!entry.metric) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "No metric");
        return;
    }

    try {
        const F32 value = std::clamp(std::any_cast<F32>(entry.metric()), 0.0f, 1.0f);
        const auto& bgColor = ImGui::ColorConvertFloat4ToU32(ctx.colorMap.at("card"));

        ImGui::PushStyleColor(ImGuiCol_FrameBg, bgColor);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImGui::GetColorU32(ImGuiCol_ButtonActive));
        ImGui::SetNextItemWidth(-1);

        ImGui::ProgressBar(value, ImVec2(0, 0), "");

        const ImVec2 rectMin = ImGui::GetItemRectMin();
        const ImVec2 rectMax = ImGui::GetItemRectMax();
        const std::string overlay = jst::fmt::format("{:.1f}%", value * 100.0f);
        const float fontSize = ImGui::GetFontSize();
        const ImVec2 textSize = ImGui::CalcTextSize(overlay.c_str());
        const F32 textX = rectMax.x - textSize.x - 6.0f * ctx.scalingFactor;
        const F32 textY = rectMin.y + (rectMax.y - rectMin.y - textSize.y) * 0.5f;
        ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(),
                                            fontSize,
                                            ImVec2(textX, textY),
                                            ImGui::GetColorU32(ImGuiCol_Text),
                                            overlay.c_str());

        ImGui::PopStyleColor(2);
    } catch (const std::bad_any_cast&) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Invalid metric type");
    }
}

// Format: `label`
static inline void RenderMetricLabel(const Block::Interface::Entry& entry,
                                     MetricContext& ctx) {
    ctx.renderHeader();

    if (!entry.metric) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "No metric");
        return;
    }

    try {
        const std::string value = std::any_cast<std::string>(entry.metric());

        const ImVec2 textSize = ImGui::CalcTextSize(value.c_str());
        const float availWidth = ImGui::GetContentRegionAvail().x;
        const float textX = ImGui::GetCursorPosX() + availWidth - textSize.x - 6.0f * ctx.scalingFactor;

        ImGui::SetCursorPosX(textX);
        ImGui::TextUnformatted(value.c_str());
    } catch (const std::bad_any_cast&) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Invalid metric type");
    }
}

// Format: `markdown`
static inline void RenderMetricMarkdown(const Block::Interface::Entry& entry,
                                        MetricContext& ctx) {
    ctx.renderHeader();

    if (!entry.metric) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "No metric");
        return;
    }

    try {
        const std::string value = std::any_cast<std::string>(entry.metric());

        ImGui::Markdown(value.c_str(), value.length(), ctx.markdownConfig);
    } catch (const std::bad_any_cast&) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Invalid metric type");
    }
}

// Format: `table`
static inline void RenderMetricTable(const Block::Interface::Entry& entry,
                                     MetricContext& ctx) {
    ctx.renderHeader();

    if (!entry.metric) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "No metric");
        return;
    }

    try {
        const std::string value = std::any_cast<std::string>(entry.metric());

        // Parse tab-delimited rows.
        std::vector<std::vector<std::string>> rows;
        std::istringstream stream(value);
        std::string line;
        while (std::getline(stream, line)) {
            if (line.empty()) continue;
            std::vector<std::string> cols;
            std::istringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, '\t')) {
                cols.push_back(cell);
            }
            if (!cols.empty()) {
                rows.push_back(std::move(cols));
            }
        }

        if (rows.empty()) {
            ImGui::TextUnformatted("No data.");
            return;
        }

        const int numCols = static_cast<int>(rows[0].size());
        const F32 rowHeight = ImGui::GetTextLineHeightWithSpacing() + 4.0f * ctx.scalingFactor;
        const F32 contentHeight = rowHeight * static_cast<F32>(rows.size());
        const F32 maxHeight = 200.0f * ctx.scalingFactor;
        const F32 tableHeight = std::min(contentHeight, maxHeight);

        if (ImGui::BeginTable("##metric_table", numCols,
                ImGuiTableFlags_Borders |
                ImGuiTableFlags_RowBg |
                ImGuiTableFlags_SizingStretchProp |
                ImGuiTableFlags_ScrollY,
                ImVec2(0.0f, tableHeight))) {
            // Header row.
            for (const auto& col : rows[0]) {
                ImGui::TableSetupColumn(col.c_str());
            }
            ImGui::TableHeadersRow();

            // Data rows.
            for (size_t r = 1; r < rows.size(); ++r) {
                ImGui::TableNextRow();
                for (size_t c = 0; c < rows[r].size(); ++c) {
                    ImGui::TableSetColumnIndex(static_cast<int>(c));
                    ImGui::TextUnformatted(rows[r][c].c_str());
                }
            }

            ImGui::EndTable();
        }
    } catch (const std::bad_any_cast&) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Invalid metric type");
    }
}

using MetricRenderer = void(*)(const Block::Interface::Entry&, MetricContext&);

static const std::unordered_map<std::string, MetricRenderer> metricRenderers = {
    {"progressbar", RenderMetricProgressBar},
    {"label",       RenderMetricLabel},
    {"markdown",    RenderMetricMarkdown},
    {"table",       RenderMetricTable},
};

struct NodeMeta {
    F32 x = 0.0f;
    F32 y = 0.0f;
    F32 width = 0.0f;
    F32 height = 0.0f;

    JST_SERDES(x, y, width, height);
};

struct SurfaceMeta {
    U64 attachedWidth = 512;
    U64 attachedHeight = 512;
    U64 detachedWidth = 512;
    U64 detachedHeight = 512;
    bool detached = false;

    JST_SERDES(attachedWidth, attachedHeight, detachedWidth, detachedHeight, detached);
};

class DefaultCompositor : public Compositor::Impl {
 public:
    Result create();
    Result destroy();

    Result present();
    Result poll();

 private:
    // Modal variables.

    enum class InterfaceModalContent : I32 {
        About          = 0,
        FlowgraphExamples = 1,
        FlowgraphInfo  = 2,
        FlowgraphClose = 3,
        RenameBlock    = 4,
        License        = 5,
        ThirdParty     = 6,
        Benchmark      = 7,
        RemoteStreaming = 8,
    };

    // ImGui variables.

    void ImGuiLoadFonts();
    void ImGuiStyleSetup();
    void ImGuiStyleScale();

    ImFont* bodyFont = nullptr;
    ImFont* h1Font = nullptr;
    ImFont* h2Font = nullptr;
    ImFont* boldFont = nullptr;

    // ImNodes variables.

    void ImNodesStyleSetup();
    void ImNodesStyleScale();

    std::unordered_map<std::string, ImNodesContext*> contexts;

    // ImGuiMarkdown variables.

    void ImGuiMarkdownStyleSetup();
    static void ImGuiMarkdownLinkCallback(ImGui::MarkdownLinkCallbackData data);
    static void ImGuiMarkdownFormatCallback(const ImGui::MarkdownFormatInfo& mdInfo, bool start);

    ImGui::MarkdownConfig markdownConfig;

    // Theme variables.

    std::string currentThemeKey = "Dark";
    static const std::unordered_map<std::string, std::unordered_map<std::string, ImVec4>> themes;
    std::unordered_map<std::string, ImVec4> ColorMap;

    // Dockspace variables.

    struct Particle {
        ImVec2 position;
        F32 velocity;
        F32 radius;
        F32 alpha;
        F32 phase;
    };
    std::vector<Particle> particles;

    // Mail variables.

    struct MailNewFlowgraph {};

    struct MailCloseFlowgraph {
        std::string flowgraph;
    };

    struct MailSaveFlowgraph {
        std::string flowgraph;
        std::string path;
    };

    struct MailResetFlowgraph {
        std::string flowgraph;
    };

    struct MailOpenFlowgraphPath {
        std::string path;
    };

    struct MailOpenFlowgraphBlob {
        std::vector<char> blob;
    };

    struct MailCreateBlock {
        std::string flowgraph;
        std::string moduleId;
        std::optional<ImVec2> gridPosition;
        DeviceType device;
        RuntimeType runtime;
        ProviderType provider;
    };

    struct MailRenameBlock {
        std::string flowgraph;
        std::string oldId;
        std::string newId;
    };

    struct MailDeleteBlock {
        std::string flowgraph;
        std::string blockId;
    };

    struct MailReloadBlock {
        std::string flowgraph;
        std::string blockId;
    };

    struct MailConnectBlock {
        std::string flowgraph;
        std::string blockName;
        std::string inputPort;
        std::string sourceBlock;
        std::string sourcePort;
    };

    struct MailDisconnectBlock {
        std::string flowgraph;
        std::string blockName;
        std::string inputPort;
    };

    struct MailReconfigureBlock {
        std::string flowgraph;
        std::string blockId;
        Parser::Map config;
        bool silent = false;
    };

    struct MailCopyBlock {
        std::string flowgraph;
        std::string blockId;
    };

    struct MailPasteBlock {
        std::string flowgraph;
        std::optional<ImVec2> gridPosition;
    };

    using Mail = std::variant<MailNewFlowgraph,
                              MailCreateBlock,
                              MailCloseFlowgraph,
                              MailSaveFlowgraph,
                              MailOpenFlowgraphPath,
                              MailOpenFlowgraphBlob,
                              MailResetFlowgraph,
                              MailRenameBlock,
                              MailDeleteBlock,
                              MailReloadBlock,
                              MailConnectBlock,
                              MailDisconnectBlock,
                              MailReconfigureBlock,
                              MailCopyBlock,
                              MailPasteBlock>;

    std::deque<Mail> mailbox;

    void enqueue(Mail&& mail);

    // Render elements.

    Result renderInfoHud();
    Result renderFullscreenHud();
    Result renderWelcomeHud();
    Result renderRemoteHud();
    Result renderNotifications();
    Result renderMenubar();
    Result renderToolbar();
    Result renderFlowgraph();

    Result renderGlobalModal();
    Result renderSeparator();
    Result renderDockspace();
    Result renderStacks();
    Result renderBackground();
    Result renderDebugLatency();
    Result renderDebugViewport();
    Result renderDebugDemo();
    Result renderDetachedSurfaces();
    Result renderDocumentations();

    // Helper functions.

    Result helperOpenFlowgraph();
    Result helperSaveFlowgraph(const std::string& flowgraph);
    Result helperCloseFlowgraph(const std::string& flowgraph);

    // Helper elements.

    void helperRenderLoadingBar(const ImVec4& color, F32 radius);
    void helperSurfaceResize(const std::shared_ptr<Module::Surface>& surface,
                             U64 width,
                             U64 height,
                             F32 scalingFactor,
                             bool detached = false);
    bool helperCheckSurfaceResize(const std::shared_ptr<Module::Surface>& surface,
                                  const SurfaceManifest& manifest,
                                  const ImVec2& availableRegion,
                                  U64& storedWidth,
                                  U64& storedHeight,
                                  F32 scalingFactor,
                                  bool detached = false);
    bool helperRenderSurfaceContent(const std::shared_ptr<Module::Surface>& surface,
                                    const SurfaceManifest& manifest,
                                    const ImVec2& surfaceSize,
                                    float rounding,
                                    bool forwardMouseEvents);

    // State

    bool infoPanelEnabled = true;
    bool debugEnableTrace = false;
    bool debugLatencyEnabled = false;
    bool debugViewportEnabled = false;
    bool debugDemoEnabled = false;
    bool debugRuntimeMetricsEnabled = false;

    bool flowgraphEnabled = true;
    bool fullscreenEnabled = false;

    bool benchmarkRunning = false;
    std::future<void> benchmarkFuture;
    std::stringstream benchmarkOutput;

    std::optional<std::string> renameBlockOldName;

    std::optional<InterfaceModalContent> globalModalContent;
    std::optional<std::string> focusedFlowgraph;
    std::deque<std::string> focusHistory;

    std::unordered_map<std::string, std::pair<bool, ImGuiID>> stacks;
    std::unordered_map<std::string, std::shared_ptr<Flowgraph>> flowgraphs;
    std::unordered_map<std::string, std::unordered_map<std::string, std::shared_ptr<Block>>> blocksCache;
    std::unordered_map<std::string, bool> openDocumentations;
    std::unordered_map<int, ImVec2> nodeSizes;

    ImGuiID mainDockspaceID = 0;

    F32 currentHeight = 0.0f;
    F32 scalingFactor = 1.0f;

    struct BlockPickerState {
        bool active = false;
        std::string flowgraphId;
        ImVec2 screenPosition;
        ImVec2 gridPosition;
        char searchBuffer[128] = "";
        int selectedIndex = 0;
        int deviceIndex = 0;
        int runtimeIndex = 0;
        int providerIndex = 0;
    };
    BlockPickerState blockPicker;

    // Clipboard state for copy/paste.
    struct ClipboardState {
        std::string moduleType;
        DeviceType device;
        RuntimeType runtime;
        ProviderType provider;
        Parser::Map config;
        bool hasData = false;
    };
    ClipboardState clipboard;

    std::string cachedQrUrl;
    std::vector<U8> cachedQrData;
    int cachedQrWidth = 0;

    // Remote streaming configuration state
    std::string remoteBrokerUrl = "https://cyberether.org";
    Instance::Remote::CodecType remoteCodec = Instance::Remote::CodecType::H264;
    Instance::Remote::EncoderType remoteEncoder = Instance::Remote::EncoderType::Auto;
    bool remoteAutoJoinSessions = false;
    U32 remoteFramerate = 30;
};

Result DefaultCompositor::create() {
    JST_INFO("[COMPOSITOR_IMPL_DEFAULT] Creating compositor.");

    // Setup theme

    ColorMap = themes.at(currentThemeKey);

    // Load example flowgraphs.

    std::vector<Registry::FlowgraphRegistration> manifest;
    const auto res = Resources::GetDefaultManifest(manifest);
    if (res == Result::SUCCESS) {
        for (const auto& entry : manifest) {
            if (Registry::RegisterFlowgraph(entry.key, entry) != Result::SUCCESS) {
                JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Failed to register flowgraph '{}'.", entry.key);
            }
        }
    } else {
        JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Failed to load default flowgraph manifest.");
    }

    // Setup ImGui

    ImGuiLoadFonts();
    ImGuiStyleSetup();
    ImGuiStyleScale();

    // Setup ImGuiMarkdown

    ImGuiMarkdownStyleSetup();

    return Result::SUCCESS;
}

Result DefaultCompositor::destroy() {
    JST_INFO("[COMPOSITOR_IMPL_DEFAULT] Destroying compositor.");

    return Result::SUCCESS;
}

Result DefaultCompositor::present() {
    // Setup frame configuration.

    if (render->scalingFactor() != scalingFactor) {
        ImGuiStyleScale();
        ImNodesStyleScale();
        scalingFactor = render->scalingFactor();
    }
    currentHeight = 0.0f;

#ifdef JST_OS_BROWSER
    if (Platform::IsFilePending()) {
        JST_CHECK(renderBackground());
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                                       ImGuiWindowFlags_NoResize |
                                       ImGuiWindowFlags_NoMove |
                                       ImGuiWindowFlags_NoBackground |
                                       ImGuiWindowFlags_AlwaysAutoResize |
                                       ImGuiWindowFlags_NoSavedSettings;
        ImGui::Begin("##file_pending", nullptr, flags);
        ImGui::Text("Select a file from the browser dialog.");
        ImGui::End();
        return Result::SUCCESS;
    }
#endif

    // Focused flowgraph heuristics.

    if (flowgraphs.empty()) {
        focusedFlowgraph.reset();
        focusHistory.clear();
    } else if (focusedFlowgraph.has_value() &&
               !flowgraphs.contains(focusedFlowgraph.value())) {
        while (!focusHistory.empty()) {
            std::string candidate = focusHistory.front();
            focusHistory.pop_front();
            if (flowgraphs.contains(candidate)) {
                focusedFlowgraph = candidate;
                break;
            }
        }
        if (!focusedFlowgraph.has_value() || !flowgraphs.contains(focusedFlowgraph.value())) {
            focusedFlowgraph = flowgraphs.begin()->first;
        }
    }

    // Update and cleanup ImNode context.

    for (auto& [flowgraphId, flowgraph] : flowgraphs) {
        if (!contexts.contains(flowgraphId)) {
            contexts[flowgraphId] = ImNodes::CreateContext();

            ImNodes::SetCurrentContext(contexts[flowgraphId]);
            ImNodesStyleSetup();
            ImNodesStyleScale();
        }
    }

    if (flowgraphs.size() != contexts.size()) {
        std::vector<std::string> toRemove;
        for (const auto& [contextId, context] : contexts) {
            if (!flowgraphs.contains(contextId)) {
                toRemove.push_back(contextId);
            }
        }
        for (const auto& contextId : toRemove) {
            ImNodes::DestroyContext(contexts[contextId]);
            contexts.erase(contextId);
        }
    }

    // Process shortcuts.

    ImGuiIO& io = ImGui::GetIO();
    const bool commandPressed = (io.KeyMods & ImGuiMod_Super) != 0;
    const bool controlPressed = (io.KeyMods & ImGuiMod_Ctrl) != 0;

    if ((commandPressed || controlPressed) && ImGui::IsKeyPressed(ImGuiKey_N, false)) {
        enqueue(MailNewFlowgraph{});
    }

    if ((commandPressed || controlPressed) && ImGui::IsKeyPressed(ImGuiKey_O, false)) {
        JST_CHECK(helperOpenFlowgraph());
    }

    if ((commandPressed || controlPressed) && ImGui::IsKeyPressed(ImGuiKey_S, false)) {
        if (focusedFlowgraph.has_value()) {
            JST_CHECK(helperSaveFlowgraph(focusedFlowgraph.value()));
        } else {
            ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "No focused flowgraph to save." });
        }
    }

    if ((commandPressed || controlPressed) && ImGui::IsKeyPressed(ImGuiKey_W, false)) {
        if (focusedFlowgraph.has_value()) {
            JST_CHECK(helperCloseFlowgraph(focusedFlowgraph.value()));
        } else {
            ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "No focused flowgraph to close." });
        }
    }

    if ((commandPressed || controlPressed) && ImGui::IsKeyPressed(ImGuiKey_I, false)) {
        if (focusedFlowgraph.has_value()) {
            globalModalContent = InterfaceModalContent::FlowgraphInfo;
        } else {
            ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "No focused flowgraph to display information." });
        }
    }

    // Render interface elements.

    JST_CHECK(renderBackground());
    JST_CHECK(renderInfoHud());
    JST_CHECK(renderFullscreenHud());
    JST_CHECK(renderWelcomeHud());
    JST_CHECK(renderRemoteHud());
    JST_CHECK(renderNotifications());
    JST_CHECK(renderDebugLatency());
    JST_CHECK(renderDebugViewport());
    JST_CHECK(renderDebugDemo());

    JST_CHECK(renderMenubar());
    JST_CHECK(renderToolbar());
    JST_CHECK(renderSeparator());
    JST_CHECK(renderDockspace());
    JST_CHECK(renderStacks());
    JST_CHECK(renderDetachedSurfaces());
    JST_CHECK(renderDocumentations());

    JST_CHECK(renderFlowgraph());
    JST_CHECK(renderGlobalModal());

    return Result::SUCCESS;
}

Result DefaultCompositor::poll() {
    // Swap the mailbox with the pending queue.

    std::deque<Mail> pending;
    pending.swap(mailbox);

    // Refresh flowgraph list and cache blocks.

    JST_CHECK(instance->flowgraphList(flowgraphs));

    blocksCache.clear();
    for (const auto& [flowgraphId, flowgraph] : flowgraphs) {
        blocksCache[flowgraphId] = flowgraph->blockList();
    }

    // Register helper functions.

    const auto& generateRandomName = [&](){
        static const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        std::string result = "f_";
        for (int i = 0; i < 8; ++i) {
            result += chars[rand() % chars.length()];
        }
        return result;
    };

    // Process queue.

    for (const auto& mail : pending) {
        if (!instance) {
            continue;
        }

        JST_CHECK(std::visit(Overloaded{
            [&](const MailNewFlowgraph&) -> Result {
                auto inst = instance;
                auto name = generateRandomName();

                Compositor::Impl::enqueue([inst, name]() -> Result {
                    std::shared_ptr<Flowgraph> flowgraph;
                    JST_CHECK(inst->flowgraphCreate(name, {}, flowgraph));
                    return Result::SUCCESS;
                });

                return Result::SUCCESS;
            },
            [&](const MailOpenFlowgraphPath& msg) -> Result {
                if (msg.path.empty()) {
                    JST_ERROR("Failed to open flowgraph from path due to empty path.");
                    return Result::ERROR;
                }

                if (!std::filesystem::exists(msg.path)) {
                    JST_ERROR("Failed to open flowgraph from path because file does not exist.");
                    return Result::ERROR;
                }

                auto inst = instance;
                auto name = generateRandomName();
                auto path = msg.path;

                Compositor::Impl::enqueue([inst, name, path]() -> Result {
                    std::shared_ptr<Flowgraph> flowgraph;
                    JST_CHECK(inst->flowgraphCreate(name, {}, flowgraph));
                    JST_CHECK(flowgraph->importFromFile(path));
                    return Result::SUCCESS;
                });

                focusedFlowgraph = name;
                return Result::SUCCESS;
            },
            [&](const MailOpenFlowgraphBlob& msg) -> Result {
                if (msg.blob.empty()) {
                    JST_ERROR("Failed to open flowgraph from blob due to invalid blob data.");
                    return Result::ERROR;
                }

                auto inst = instance;
                auto name = generateRandomName();
                auto blob = msg.blob;

                Compositor::Impl::enqueue([inst, name, blob]() -> Result {
                    std::shared_ptr<Flowgraph> flowgraph;
                    JST_CHECK(inst->flowgraphCreate(name, {}, flowgraph));
                    JST_CHECK(flowgraph->importFromBlob(blob));
                    return Result::SUCCESS;
                });

                focusedFlowgraph = name;
                return Result::SUCCESS;
            },
            [&](const MailSaveFlowgraph& msg) -> Result {
                if (!flowgraphs.contains(msg.flowgraph)) {
                    JST_ERROR("Failed to save flowgraph because flowgraph was not found.");
                    return Result::ERROR;
                }

                if (msg.path.empty()) {
                    JST_ERROR("Failed to save flowgraph due to empty path.");
                    return Result::ERROR;
                }

                auto flowgraph = flowgraphs[msg.flowgraph];
                auto path = msg.path;

                Compositor::Impl::enqueue([flowgraph, path]() -> Result {
                    JST_CHECK(flowgraph->exportToFile(path));
                    return Result::SUCCESS;
                });

                return Result::SUCCESS;
            },
            [&](const MailResetFlowgraph&) -> Result {
                // TODO: Implement.
                return Result::SUCCESS;
            },
            [&](const MailCloseFlowgraph& msg) -> Result {
                if (!flowgraphs.contains(msg.flowgraph)) {
                    JST_ERROR("Failed to close flowgraph because flowgraph was not found.");
                    return Result::ERROR;
                }

                auto inst = instance;
                auto name = msg.flowgraph;

                Compositor::Impl::enqueue([inst, name]() -> Result {
                    JST_CHECK(inst->flowgraphDestroy(name));
                    return Result::SUCCESS;
                });

                return Result::SUCCESS;
            },
            [&](const MailCreateBlock& msg) -> Result {
                if (!flowgraphs.contains(msg.flowgraph)) {
                    JST_ERROR("Failed to create block because flowgraph was not found.");
                    return Result::ERROR;
                }

                auto flowgraph = flowgraphs[msg.flowgraph];
                const auto& existingBlocks = blocksCache.at(msg.flowgraph);

                std::string baseName = msg.moduleId;
                std::string blockName = baseName;
                int suffix = 1;
                while (existingBlocks.contains(blockName)) {
                    blockName = jst::fmt::format("{}_{}", baseName, suffix++);
                }

                if (msg.gridPosition.has_value()) {
                    const auto& pos = msg.gridPosition.value();
                    flowgraph->setMeta("node", NodeMeta{pos.x, pos.y, 140.0f, 0.0f}, blockName);
                }

                Compositor::Impl::enqueue([flowgraph, blockName, msg]() -> Result {
                    JST_CHECK_ALLOW(flowgraph->blockCreate(blockName,
                                                          msg.moduleId,
                                                          {},
                                                          {},
                                                          msg.device,
                                                          msg.runtime,
                                                          msg.provider),
                                    Result::INCOMPLETE);

                    return Result::SUCCESS;
                });

                return Result::SUCCESS;
            },
            [&](const MailRenameBlock&) -> Result {
                // TODO: Implement.
                return Result::SUCCESS;
            },
            [&](const MailDeleteBlock& msg) -> Result {
                if (!flowgraphs.contains(msg.flowgraph)) {
                    JST_ERROR("Failed to delete block because flowgraph was not found.");
                    return Result::ERROR;
                }

                if (msg.blockId.empty()) {
                    JST_ERROR("Failed to delete block due to empty block ID.");
                    return Result::ERROR;
                }

                auto flowgraph = flowgraphs[msg.flowgraph];
                auto blockId = msg.blockId;

                Compositor::Impl::enqueue([flowgraph, blockId]() -> Result {
                    return flowgraph->blockDestroy(blockId);
                });

                return Result::SUCCESS;
            },
            [&](const MailReloadBlock& msg) -> Result {
                if (!flowgraphs.contains(msg.flowgraph)) {
                    JST_ERROR("Failed to reload block because flowgraph was not found.");
                    return Result::ERROR;
                }

                if (msg.blockId.empty()) {
                    JST_ERROR("Failed to reload block due to empty block ID.");
                    return Result::ERROR;
                }

                auto flowgraph = flowgraphs[msg.flowgraph];
                auto blockId = msg.blockId;

                Compositor::Impl::enqueue([flowgraph, blockId]() -> Result {
                    Parser::Map config;
                    JST_CHECK(flowgraph->blockConfig(blockId, config));
                    return flowgraph->blockRecreate(blockId, config);
                });

                return Result::SUCCESS;
            },
            [&](const MailConnectBlock& msg) -> Result {
                if (!flowgraphs.contains(msg.flowgraph)) {
                    JST_ERROR("Failed to connect block because flowgraph was not found.");
                    return Result::ERROR;
                }

                auto flowgraph = flowgraphs[msg.flowgraph];
                auto blockName = msg.blockName;
                auto inputPort = msg.inputPort;
                auto sourceBlock = msg.sourceBlock;
                auto sourcePort = msg.sourcePort;

                Compositor::Impl::enqueue([flowgraph, blockName, inputPort, sourceBlock, sourcePort]() -> Result {
                    return flowgraph->blockConnect(blockName, inputPort, sourceBlock, sourcePort);
                });

                return Result::SUCCESS;
            },
            [&](const MailDisconnectBlock& msg) -> Result {
                if (!flowgraphs.contains(msg.flowgraph)) {
                    JST_ERROR("Failed to disconnect block because flowgraph was not found.");
                    return Result::ERROR;
                }

                auto flowgraph = flowgraphs[msg.flowgraph];
                auto blockName = msg.blockName;
                auto inputPort = msg.inputPort;

                Compositor::Impl::enqueue([flowgraph, blockName, inputPort]() -> Result {
                    return flowgraph->blockDisconnect(blockName, inputPort);
                });

                return Result::SUCCESS;
            },
            [&](const MailReconfigureBlock& msg) -> Result {
                if (!flowgraphs.contains(msg.flowgraph)) {
                    JST_ERROR("Failed to reconfigure block because flowgraph was not found.");
                    return Result::ERROR;
                }

                auto flowgraph = flowgraphs[msg.flowgraph];
                auto blockId = msg.blockId;
                auto config = msg.config;

                Compositor::Impl::enqueue([flowgraph, blockId, config]() -> Result {
                    return flowgraph->blockReconfigure(blockId, config);
                }, msg.silent);

                return Result::SUCCESS;
            },
            [&](const MailCopyBlock& msg) -> Result {
                if (!flowgraphs.contains(msg.flowgraph)) {
                    JST_ERROR("Failed to copy block because flowgraph was not found.");
                    return Result::ERROR;
                }

                if (msg.blockId.empty()) {
                    JST_ERROR("Failed to copy block due to empty block ID.");
                    return Result::ERROR;
                }

                if (!blocksCache.at(msg.flowgraph).contains(msg.blockId)) {
                    JST_ERROR("Failed to copy block because block was not found.");
                    return Result::ERROR;
                }

                auto flowgraph = flowgraphs[msg.flowgraph];
                const auto& block = blocksCache.at(msg.flowgraph).at(msg.blockId);

                clipboard.moduleType = block->config().type();
                clipboard.device = block->device();
                clipboard.runtime = block->runtime();
                clipboard.provider = block->provider();
                flowgraph->blockConfig(msg.blockId, clipboard.config);
                clipboard.hasData = true;

                ImGui::InsertNotification({ ImGuiToastType_Info, 3000, "Block copied to clipboard." });

                return Result::SUCCESS;
            },
            [&](const MailPasteBlock& msg) -> Result {
                if (!flowgraphs.contains(msg.flowgraph)) {
                    JST_ERROR("Failed to paste block because flowgraph was not found.");
                    return Result::ERROR;
                }

                if (!clipboard.hasData) {
                    ImGui::InsertNotification({ ImGuiToastType_Warning, 3000, "Clipboard is empty." });
                    return Result::SUCCESS;
                }

                auto flowgraph = flowgraphs[msg.flowgraph];
                const auto& existingBlocks = blocksCache.at(msg.flowgraph);

                // Generate unique name.

                std::string baseName = clipboard.moduleType;
                std::string blockName = baseName;
                int suffix = 1;
                while (existingBlocks.contains(blockName)) {
                    blockName = jst::fmt::format("{}_{}", baseName, suffix++);
                }

                // Set grid position if provided.

                if (msg.gridPosition.has_value()) {
                    const auto& pos = msg.gridPosition.value();
                    flowgraph->setMeta("node", NodeMeta{pos.x, pos.y, 140.0f, 0.0f}, blockName);
                }

                auto config = clipboard.config;
                auto moduleType = clipboard.moduleType;
                auto device = clipboard.device;
                auto runtime = clipboard.runtime;
                auto provider = clipboard.provider;

                Compositor::Impl::enqueue([flowgraph, blockName, moduleType, config, device, runtime, provider]() -> Result {
                    JST_CHECK_ALLOW(flowgraph->blockCreate(blockName,
                                                          moduleType,
                                                          config,
                                                          {},
                                                          device,
                                                          runtime,
                                                          provider),
                                    Result::INCOMPLETE);

                    return Result::SUCCESS;
                });

                return Result::SUCCESS;
            }
        }, mail));
    }

    Command completed;
    while (dequeue(completed)) {
        if (completed.silent) {
            continue;
        }
        NotifyResultClean(completed.result, completed.message);
    }

    return Result::SUCCESS;
}

void DefaultCompositor::enqueue(Mail&& mail) {
    mailbox.emplace_back(std::move(mail));
}

//
// Flowgraph Render
//

Result DefaultCompositor::renderFlowgraph() {
    if (!flowgraphEnabled || fullscreenEnabled) {
        return Result::SUCCESS;
    }

    // Draw windows.

    for (const auto& [flowgraphId, flowgraph] : flowgraphs) {
        JST_ASSERT(flowgraph != nullptr, "[COMPOSITOR] Flowgraph '{}' is null.", flowgraphId);
        JST_ASSERT(contexts.contains(flowgraphId), "[COMPOSITOR] Context for '{}' not found.", flowgraphId);
        JST_ASSERT(blocksCache.contains(flowgraphId), "[COMPOSITOR] BlocksCache for '{}' not found.", flowgraphId);

        ImGuiWindowClass windowClass;
        windowClass.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoCloseButton | ImGuiDockNodeFlags_NoWindowMenuButton;
        ImGui::SetNextWindowClass(&windowClass);

        bool flowgraphOpen = true;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f * scalingFactor, 8.0f * scalingFactor));
        ImGui::SetNextWindowSize(ImVec2(640.0f * scalingFactor, 480.0f * scalingFactor), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowDockID(mainDockspaceID, ImGuiCond_FirstUseEver);
        const std::string windowTitle = flowgraph->title().empty() ? flowgraphId : flowgraph->title();
        const bool windowExpanded = ImGui::Begin(jst::fmt::format("{}###{}", windowTitle, flowgraphId).c_str(), &flowgraphOpen);
        ImGui::PopStyleVar();

        if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows)) {
            if (!focusedFlowgraph.has_value() || focusedFlowgraph.value() != flowgraphId) {
                if (focusedFlowgraph.has_value()) {
                    focusHistory.push_front(focusedFlowgraph.value());
                    if (focusHistory.size() > 50) {
                        focusHistory.pop_back();
                    }
                }
                focusedFlowgraph = flowgraphId;
            }
        }

        if (!flowgraphOpen) {
            if (flowgraph->path().empty()) {
                focusedFlowgraph = flowgraphId;
                globalModalContent = InterfaceModalContent::FlowgraphClose;
            } else {
                enqueue(MailCloseFlowgraph{flowgraphId});
            }
            ImGui::End();
            continue;
        }

        if (!windowExpanded) {
            ImGui::End();
            continue;
        }

        auto idFromStr = [](const std::string& s) {
            return static_cast<int>(std::hash<std::string>{}(s));
        };

        ImGui::PushFont(ImGui::GetFont(), ImGui::GetFontSize() * 0.90f);
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 12.0f * scalingFactor);

        ImNodes::SetCurrentContext(contexts.at(flowgraphId));
        ImNodes::BeginNodeEditor();

        struct PinInfo {
            std::string block;
            std::string port;
            bool isInput;
        };

        std::unordered_map<int, PinInfo> pinIdToInfo;
        std::unordered_map<int, std::string> nodeIdToBlockName;

        const auto& blocks = blocksCache.at(flowgraphId);

        for (const auto& [blockName, blockPtr] : blocks) {
            if (!blockPtr ||
                !blockPtr->interface() ||
                blockPtr->state() == Block::State::Destroying ||
                blockPtr->state() == Block::State::Destroyed) {
                continue;
            }

            const int nodeId = idFromStr("node:" + flowgraphId + ":" + blockName);
            nodeIdToBlockName[nodeId] = blockName;
            if (!nodeSizes.contains(nodeId)) {
                nodeSizes[nodeId] = ImVec2(140.0f * scalingFactor, 0.0f);
            }
            NodeMeta nodeMeta;
            if (flowgraph->getMeta("node", nodeMeta, blockName) == Result::SUCCESS) {
                ImNodes::SetNodeGridSpacePos(nodeId, ImVec2(nodeMeta.x, nodeMeta.y));
                if (nodeMeta.width > 0.0f) {
                    nodeSizes[nodeId].x = nodeMeta.width * scalingFactor;
                }
                if (nodeMeta.height > 0.0f) {
                    nodeSizes[nodeId].y = nodeMeta.height * scalingFactor;
                }
            }

            bool hasSurfaces = false;
            bool isDetached = false;
            const U64 defaultAttachedSurfaceHeight = SurfaceMeta{}.attachedHeight;
            U64 attachedSurfaceHeight = defaultAttachedSurfaceHeight;

            if (!blockPtr->surfaces().empty()) {
                const auto& surface = blockPtr->surfaces().front();
                if (!surface->manifests().empty()) {
                    hasSurfaces = true;
                    const auto& manifest = surface->manifests().front();
                    SurfaceMeta surfaceMeta;
                    flowgraph->getMeta("surface_" + manifest.id, surfaceMeta, blockName);
                    attachedSurfaceHeight = surfaceMeta.attachedHeight;
                    isDetached = surfaceMeta.detached;
                }
            }

            ImNodes::SetNodeVerticalResizeEnabled(nodeId, hasSurfaces && !isDetached);

            const Block::State blockState = blockPtr->state();
            const bool isReloading = blockState == Block::State::Creating ||
                                     blockState == Block::State::Incomplete;
            const bool shouldResetNodeHeight = (!hasSurfaces || isDetached) && !isReloading;
            const F32 minimumSurfaceHeight = static_cast<F32>(attachedSurfaceHeight) * scalingFactor;
            const bool shouldSeedSurfaceHeight = hasSurfaces && !isDetached &&
                                                 attachedSurfaceHeight == defaultAttachedSurfaceHeight &&
                                                 nodeSizes[nodeId].y < minimumSurfaceHeight;
            const bool hasValidNodeHeight = nodeSizes[nodeId].y > 0.0f;

            if (shouldResetNodeHeight || (isReloading && !hasValidNodeHeight)) {
                nodeSizes[nodeId].y = 0.0f;
            } else if (shouldSeedSurfaceHeight) {
                nodeSizes[nodeId].y = minimumSurfaceHeight;
            }
            ImNodes::SetNodeDimensions(nodeId, nodeSizes[nodeId]);

            if (blockPtr->state() == Block::State::Errored) {
                const ImU32 stateColor = ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_outline_error"));
                ImNodes::PushColorStyle(ImNodesCol_NodeOutline, stateColor);
                ImNodes::PushColorStyle(ImNodesCol_Pin, stateColor);
                ImNodes::PushColorStyle(ImNodesCol_PinHovered, stateColor);
            } else if (blockPtr->state() == Block::State::Creating || blockPtr->state() == Block::State::Incomplete) {
                const ImU32 stateColor = ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_outline_pending"));
                ImNodes::PushColorStyle(ImNodesCol_NodeOutline, stateColor);
                ImNodes::PushColorStyle(ImNodesCol_Pin, stateColor);
                ImNodes::PushColorStyle(ImNodesCol_PinHovered, stateColor);
            }

            ImNodes::BeginNode(nodeId);

            // Draw block title.

            {
                ImNodes::BeginNodeTitleBar();

                const bool hasDiagnostic = (blockPtr->state() == Block::State::Errored ||
                                            blockPtr->state() == Block::State::Incomplete) &&
                                           !blockPtr->diagnostic().empty();

                ImGui::PushFont(ImGui::GetFont(), ImGui::GetFontSize() * 1.15f);
                const float textWidth = ImGui::CalcTextSize(blockPtr->config().title().c_str()).x;
                const float availWidth = ImGui::GetContentRegionAvail().x;
                const float titleStartX = ImGui::GetCursorPosX();
                ImGui::SetCursorPosX(titleStartX + ImMax(0.0f, (availWidth - textWidth) * 0.5f));
                ImGui::TextUnformatted(blockPtr->config().title().c_str());
                ImGui::PopFont();

                if (hasDiagnostic) {
                    ImGui::SameLine();
                    ImGui::SetCursorPosX(titleStartX + availWidth - ImGui::CalcTextSize(ICON_FA_SKULL).x);
                    const ImU32 iconColor = (blockPtr->state() == Block::State::Errored)
                        ? ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_outline_error"))
                        : ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_outline_pending"));
                    ImGui::PushStyleColor(ImGuiCol_Text, iconColor);
                    ImGui::TextUnformatted(ICON_FA_SKULL);
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                        const auto diagnosticMessage = CleanUserMessage(blockPtr->diagnostic());
                        ImGui::BeginTooltip();
                        ImGui::PushStyleColor(ImGuiCol_Text, iconColor);
                        ImGui::TextUnformatted(ICON_FA_TRIANGLE_EXCLAMATION);
                        ImGui::PopStyleColor();
                        ImGui::SameLine();
                        ImGui::TextUnformatted("Diagnostic");
                        ImGui::Separator();
                        ImGui::TextUnformatted(diagnosticMessage.c_str());
                        ImGui::EndTooltip();
                    }
                    ImGui::PopStyleColor();
                }

                ImNodes::EndNodeTitleBar();
            }

            // Draw block name.

            {
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 12.0f * scalingFactor);
                ImGui::PushFont(ImGui::GetFont(), ImGui::GetFontSize() * 0.75f);
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertFloat4ToU32(ColorMap.at("text_secondary")));
                const float idWidth = ImGui::CalcTextSize(blockName.c_str()).x;
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImMax(0.0f, (ImGui::GetContentRegionAvail().x - idWidth) * 0.5f));
                ImGui::TextUnformatted(blockName.c_str());
                ImGui::PopStyleColor();
                ImGui::PopFont();
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 8.0f * scalingFactor);
            }

            for (const auto& [slot, interfaceInfo] : blockPtr->interface()->inputs()) {
                const int pinId = idFromStr("pin:in:" + flowgraphId + ":" + blockName + ":" + slot);
                pinIdToInfo[pinId] = {blockName, slot, true};

                ImNodes::PushAttributeFlag(ImNodesAttributeFlags_EnableLinkDetachWithDragClick);
                ImNodes::BeginInputAttribute(pinId, ImNodesPinShape_CircleFilled);
                ImGui::TextUnformatted(interfaceInfo.label.c_str());
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(interfaceInfo.help.c_str());
                    ImGui::EndTooltip();
                }
                ImNodes::EndInputAttribute();
                ImNodes::PopAttributeFlag();
            }

            for (const auto& [slot, interfaceInfo] : blockPtr->interface()->outputs()) {
                const int pinId = idFromStr("pin:out:" + flowgraphId + ":" + blockName + ":" + slot);
                pinIdToInfo[pinId] = {blockName, slot, false};

                ImNodes::BeginOutputAttribute(pinId, ImNodesPinShape_CircleFilled);
                const float textWidth = ImGui::CalcTextSize(interfaceInfo.label.c_str()).x;
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImMax(0.0f, ImGui::GetContentRegionAvail().x - textWidth));
                ImGui::TextUnformatted(interfaceInfo.label.c_str());
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(interfaceInfo.help.c_str());
                    ImGui::EndTooltip();
                }
                ImNodes::EndOutputAttribute();
            }

            if (blockPtr->state() == Block::State::Creating) {
                ImGui::Spacing();
                helperRenderLoadingBar(ColorMap.at("node_outline_pending"), 4.0f * scalingFactor);
                ImGui::Spacing();
            }

            // Render metrics.

            if (blockPtr->state() != Block::State::Creating) {
                if (!blockPtr->interface()->metrics().empty()) {
                    MetricContext ctx{
                        .scalingFactor = scalingFactor,
                        .colorMap = ColorMap,
                        .label = {},
                        .help = {},
                        .markdownConfig = markdownConfig,
                    };

                    for (const auto& [name, entry] : blockPtr->interface()->metrics()) {
                        ctx.label = entry.label;
                        ctx.help = entry.help;

                        const auto parts = Parser::SplitString(entry.format, ":");
                        const auto& kind = parts[0];

                        if (kind == "markdown") {
                            const ImGuiID blockEditingId = ImGui::GetID("##multiline_editing");
                            if (ImGui::GetStateStorage()->GetBool(blockEditingId, false)) {
                                continue;
                            }
                        }

                        if (!metricRenderers.contains(kind)) {
                            JST_ERROR("[COMPOSITOR_IMPL_DEFAULT] Unknown metric type '{}' for metric '{}'", kind, name);
                            continue;
                        }

                        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f * scalingFactor, 2.0f * scalingFactor));
                        metricRenderers.at(kind)(entry, ctx);
                        ImGui::PopStyleVar();
                    }

                    ImGui::Spacing();
                }
            }

            // Render configs.

            if (blockPtr->state() != Block::State::Creating) {
                Parser::Map cfg;
                if (blockPtr->config(cfg) == Result::SUCCESS && !blockPtr->interface()->configs().empty()) {
                    bool cfgChanged = false;

                    FieldContext ctx{
                        .cfg = cfg,
                        .scalingFactor = scalingFactor,
                        .colorMap = ColorMap,
                        .label = {},
                        .help = {},
                        .asyncApply = [this, flowgraphId, blockName, cfgCopy = cfg](std::string name, std::string value) mutable {
                            cfgCopy[name] = std::move(value);
                            enqueue(MailReconfigureBlock{flowgraphId, blockName, std::move(cfgCopy), false});
                        },
                    };

                    for (const auto& [name, entry] : blockPtr->interface()->configs()) {
                        ctx.label = entry.label.empty() ? name : entry.label;
                        ctx.help = entry.help;

                        const auto parts = Parser::SplitString(entry.format, ":");
                        const auto& kind = parts[0];

                        if (!fieldRenderers.contains(kind)) {
                            JST_ERROR("[COMPOSITOR_IMPL_DEFAULT] Unknown field type '{}' for config '{}'", kind, name);
                            continue;
                        }

                        std::string encoded;
                        if (cfg.contains(name)) {
                            Parser::TypedToString(cfg[name], encoded);
                        }

                        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f * scalingFactor, 2.0f * scalingFactor));
                        cfgChanged |= fieldRenderers.at(kind)(name, parts, encoded, ctx);
                        ImGui::PopStyleVar();
                    }

                    if (cfgChanged) {
                        enqueue(MailReconfigureBlock{flowgraphId, blockName, cfg, ctx.silent});
                    }
                }
            }

            // Render surfaces.

            if (blockPtr->state() != Block::State::Creating) {
                for (const auto& surface : blockPtr->surfaces()) {
                    for (const auto& manifest : surface->manifests()) {
                        if (!manifest.surface || manifest.surface->raw() == 0) {
                            continue;
                        }

                        const std::string surfaceMetaKey = "surface_" + manifest.id;
                        SurfaceMeta surfaceMeta;
                        flowgraph->getMeta(surfaceMetaKey, surfaceMeta, blockName);

                        if (surfaceMeta.detached) {
                            continue;
                        }

                        const auto availableRegion = ImGui::GetContentRegionAvail();
                        if (helperCheckSurfaceResize(surface,
                                                     manifest,
                                                     availableRegion,
                                                     surfaceMeta.attachedWidth,
                                                     surfaceMeta.attachedHeight,
                                                     scalingFactor,
                                                     false)) {
                            flowgraph->setMeta(surfaceMetaKey, surfaceMeta, blockName);
                        }

                        const ImVec2 surfaceSize(availableRegion.x, availableRegion.y);
                        const ImVec2 cursorPos = ImGui::GetCursorScreenPos();
                        const float rounding = ImGui::GetStyle().FrameRounding;

                        const bool surfaceHovered = helperRenderSurfaceContent(surface, manifest, surfaceSize, rounding, false);

                        if (surfaceHovered) {
                            const ImVec2 cursorEnd = ImVec2(cursorPos.x + surfaceSize.x, cursorPos.y + surfaceSize.y);
                            const float buttonSize = 24.0f * scalingFactor;
                            const float buttonPadding = 8.0f * scalingFactor;
                            const ImVec2 buttonPos(cursorEnd.x - buttonSize - buttonPadding, cursorPos.y + buttonPadding);

                            ImDrawList* drawList = ImGui::GetWindowDrawList();
                            const ImVec2 buttonEnd(buttonPos.x + buttonSize, buttonPos.y + buttonSize);
                            const ImVec2 mousePos = ImGui::GetMousePos();
                            const bool buttonHovered = mousePos.x >= buttonPos.x && mousePos.x <= buttonEnd.x &&
                                                       mousePos.y >= buttonPos.y && mousePos.y <= buttonEnd.y;

                            ImU32 buttonColor = IM_COL32(30, 30, 30, 200);
                            if (buttonHovered) {
                                buttonColor = IM_COL32(60, 60, 60, 230);
                                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                                    surfaceMeta.detached = true;
                                    flowgraph->setMeta(surfaceMetaKey, surfaceMeta, blockName);
                                    helperSurfaceResize(surface,
                                                        static_cast<U64>(surfaceMeta.detachedWidth * ImGui::GetIO().DisplayFramebufferScale.x),
                                                        static_cast<U64>(surfaceMeta.detachedHeight * ImGui::GetIO().DisplayFramebufferScale.y),
                                                        scalingFactor,
                                                        true);
                                }
                            }

                            drawList->AddRectFilled(buttonPos, buttonEnd, buttonColor, 4.0f * scalingFactor);

                            const char* icon = ICON_FA_UP_RIGHT_AND_DOWN_LEFT_FROM_CENTER;
                            const ImVec2 textSize = ImGui::CalcTextSize(icon);
                            const ImVec2 textPos(buttonPos.x + (buttonSize - textSize.x) * 0.5f,
                                                 buttonPos.y + (buttonSize - textSize.y) * 0.5f);
                            drawList->AddText(textPos, IM_COL32(255, 255, 255, 255), icon);
                        }
                    }
                }
            }

            ImNodes::EndNode();

            if (blockPtr->state() == Block::State::Errored ||
                blockPtr->state() == Block::State::Creating ||
                blockPtr->state() == Block::State::Incomplete) {
                ImNodes::PopColorStyle();
                ImNodes::PopColorStyle();
                ImNodes::PopColorStyle();
            }
        }

        // Draw links between connected blocks.

        struct LinkInfo {
            std::string consumerName;
            std::string inputSlot;
            std::string producerName;
            std::string producerPort;
            const TensorLink* tensorLink;
        };
        std::unordered_map<int, LinkInfo> linkIdToConnection;
        for (const auto& [consumerName, consumerPtr] : blocks) {
            if (!consumerPtr) {
                continue;
            }

            for (const auto& [inputSlot, link] : consumerPtr->inputs()) {
                const int inPinId = idFromStr("pin:in:" + flowgraphId + ":" + consumerName + ":" + inputSlot);
                const int outPinId = idFromStr("pin:out:" + flowgraphId + ":" + link.block + ":" + link.port);
                const int linkId = idFromStr("link:" + flowgraphId + ":" + consumerName + ":" + inputSlot);
                const bool linkUnresolved = !link.resolved();

                if (!pinIdToInfo.contains(inPinId) || !pinIdToInfo.contains(outPinId)) {
                    continue;
                }

                if (linkUnresolved) {
                    const ImU32 greyColor = ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_outline_pending"));
                    ImNodes::PushColorStyle(ImNodesCol_Link, greyColor);
                    ImNodes::PushColorStyle(ImNodesCol_LinkHovered, greyColor);
                    ImNodes::PushColorStyle(ImNodesCol_LinkSelected, greyColor);
                }

                linkIdToConnection[linkId] = {consumerName, inputSlot, link.block, link.port, &link};
                ImNodes::Link(linkId, outPinId, inPinId);

                if (linkUnresolved) {
                    ImNodes::PopColorStyle();
                    ImNodes::PopColorStyle();
                    ImNodes::PopColorStyle();
                }
            }
        }

        // Block picker render.

        if (blockPicker.active && blockPicker.flowgraphId == flowgraphId) {
            const ImVec2 pickerSize(280.0f * scalingFactor, 320.0f * scalingFactor);
            const ImVec2 screenPos = blockPicker.screenPosition;

            ImGui::SetNextWindowPos(screenPos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(pickerSize, ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.95f);

            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 12.0f * scalingFactor);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f * scalingFactor, 8.0f * scalingFactor));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 2.0f * scalingFactor);
            ImGui::PushStyleColor(ImGuiCol_Border, ColorMap.at("border"));

            ImGuiWindowFlags pickerFlags = ImGuiWindowFlags_NoTitleBar |
                                           ImGuiWindowFlags_NoResize |
                                           ImGuiWindowFlags_NoMove |
                                           ImGuiWindowFlags_NoSavedSettings |
                                           ImGuiWindowFlags_NoScrollbar |
                                           ImGuiWindowFlags_NoNav;

            if (ImGui::Begin("##block_picker", nullptr, pickerFlags)) {
                // Close on escape or click outside.

                if (ImGui::IsKeyPressed(ImGuiKey_Escape) ||
                    (ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
                     !ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows) &&
                     !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId))) {
                    blockPicker.active = false;
                }

                // Header bar.

               {
                    // Title.

                    ImGui::PushFont(ImGui::GetFont(), ImGui::GetFontSize() * 1.15f);
                    const float textWidth = ImGui::CalcTextSize("Block Picker").x;
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImMax(0.0f, (ImGui::GetContentRegionAvail().x - textWidth) * 0.5f));
                    ImGui::TextUnformatted("Block Picker");
                    ImGui::PopFont();

                    // Subtitle.

                    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 4.0f * scalingFactor);
                    const float subtitleFontSize = ImGui::GetFontSize() * 0.85f;
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
                    ImGui::PushFont(ImGui::GetFont(), subtitleFontSize);
                    const char* subtitle = "Use arrows to navigate, Enter to create";
                    const float subtitleWidth = ImGui::CalcTextSize(subtitle).x;
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImMax(0.0f, (ImGui::GetContentRegionAvail().x - subtitleWidth) * 0.5f));
                    ImGui::TextUnformatted(subtitle);
                    ImGui::PopFont();
                    ImGui::PopStyleColor();
                }

                ImGui::SetNextItemWidth(-1);
                if (!ImGui::IsAnyItemActive() && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId)) {
                    ImGui::SetKeyboardFocusHere();
                }
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8.0f * scalingFactor);
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f * scalingFactor, 8.0f * scalingFactor));
                ImGui::InputTextWithHint("##picker_search", "Search blocks...", blockPicker.searchBuffer, sizeof(blockPicker.searchBuffer));
                ImGui::PopStyleVar(2);

                // Build filtered block list.

                struct BlockItem {
                    std::string type;
                    std::string title;
                    std::string summary;
                };
                std::vector<BlockItem> filteredBlocks;

                const std::string query = std::string(blockPicker.searchBuffer);
                auto matches = [&](const std::string& title, const std::string& summary) -> bool {
                    if (query.empty()) return true;
                    std::string t = title, s = summary, q = query;
                    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
                    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                    std::transform(q.begin(), q.end(), q.begin(), ::tolower);
                    return (t.find(q) != std::string::npos) || (s.find(q) != std::string::npos);
                };

                for (const auto& entry : Registry::ListAvailableBlocks("")) {
                    if (matches(entry.title, entry.summary)) {
                        filteredBlocks.push_back({entry.type, entry.title, entry.summary});
                    }
                }

                const int filteredCount = static_cast<int>(filteredBlocks.size());
                if (blockPicker.selectedIndex >= filteredCount) {
                    blockPicker.selectedIndex = ImMax(0, filteredCount - 1);
                }

                // Get available implementations for the selected block.

                std::vector<Registry::ModuleRegistration> availableModules;
                if (filteredCount > 0 && blockPicker.selectedIndex >= 0) {
                    availableModules = Registry::ListAvailableModules(filteredBlocks[blockPicker.selectedIndex].type);
                }
                const int moduleCount = static_cast<int>(availableModules.size());

                // Clamp implementation index.

                if (blockPicker.deviceIndex >= moduleCount) {
                    blockPicker.deviceIndex = ImMax(0, moduleCount - 1);
                }

                // Build unique lists of devices, runtimes, and providers.

                std::vector<DeviceType> uniqueDevices;
                std::vector<RuntimeType> uniqueRuntimes;
                std::vector<ProviderType> uniqueProviders;

                for (const auto& mod : availableModules) {
                    if (std::find(uniqueDevices.begin(), uniqueDevices.end(), mod.device) == uniqueDevices.end()) {
                        uniqueDevices.push_back(mod.device);
                    }
                    if (std::find(uniqueRuntimes.begin(), uniqueRuntimes.end(), mod.runtime) == uniqueRuntimes.end()) {
                        uniqueRuntimes.push_back(mod.runtime);
                    }
                    if (std::find(uniqueProviders.begin(), uniqueProviders.end(), mod.provider) == uniqueProviders.end()) {
                        uniqueProviders.push_back(mod.provider);
                    }
                }

                // Provide defaults for composite blocks without module registrations.
                // TODO: Properly fix module-less block creation.

                if (uniqueDevices.empty()) {
                    uniqueDevices.push_back(DeviceType::CPU);
                }
                if (uniqueRuntimes.empty()) {
                    uniqueRuntimes.push_back(RuntimeType::NATIVE);
                }
                if (uniqueProviders.empty()) {
                    uniqueProviders.push_back("generic");
                }

                // Clamp indices.

                if (blockPicker.deviceIndex >= static_cast<int>(uniqueDevices.size())) {
                    blockPicker.deviceIndex = ImMax(0, static_cast<int>(uniqueDevices.size()) - 1);
                }
                if (blockPicker.runtimeIndex >= static_cast<int>(uniqueRuntimes.size())) {
                    blockPicker.runtimeIndex = ImMax(0, static_cast<int>(uniqueRuntimes.size()) - 1);
                }
                if (blockPicker.providerIndex >= static_cast<int>(uniqueProviders.size())) {
                    blockPicker.providerIndex = ImMax(0, static_cast<int>(uniqueProviders.size()) - 1);
                }

                // Display implementation selection dropdowns.

                if (!uniqueDevices.empty() && !uniqueRuntimes.empty() && !uniqueProviders.empty()) {
                    const float spacing = 4.0f * scalingFactor;
                    const float availWidth = ImGui::GetContentRegionAvail().x;
                    const float comboWidth = (availWidth - spacing * 2) / 3.0f;

                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f * scalingFactor);
                    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 6.0f * scalingFactor);
                    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f * scalingFactor, 6.0f * scalingFactor));

                    // Device dropdown.

                    ImGui::SetNextItemWidth(comboWidth);
                    std::string devicePreview = std::string(ICON_FA_MICROCHIP) + " " + GetDevicePrettyName(uniqueDevices[blockPicker.deviceIndex]);
                    if (ImGui::BeginCombo("##device", devicePreview.c_str(), ImGuiComboFlags_NoArrowButton)) {
                        for (int i = 0; i < static_cast<int>(uniqueDevices.size()); ++i) {
                            const bool selected = (i == blockPicker.deviceIndex);
                            if (ImGui::Selectable(GetDevicePrettyName(uniqueDevices[i]), selected)) {
                                blockPicker.deviceIndex = i;
                            }
                            if (selected) ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }

                    ImGui::SameLine(0, spacing);

                    // Runtime dropdown.

                    ImGui::SetNextItemWidth(comboWidth);
                    std::string runtimePreview = std::string(ICON_FA_GAUGE_HIGH) + " " + GetRuntimePrettyName(uniqueRuntimes[blockPicker.runtimeIndex]);
                    if (ImGui::BeginCombo("##runtime", runtimePreview.c_str(), ImGuiComboFlags_NoArrowButton)) {
                        for (int i = 0; i < static_cast<int>(uniqueRuntimes.size()); ++i) {
                            const bool selected = (i == blockPicker.runtimeIndex);
                            if (ImGui::Selectable(GetRuntimePrettyName(uniqueRuntimes[i]), selected)) {
                                blockPicker.runtimeIndex = i;
                            }
                            if (selected) ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }

                    ImGui::SameLine(0, spacing);

                    // Provider dropdown.

                    ImGui::SetNextItemWidth(comboWidth);
                    std::string providerPreview = std::string(ICON_FA_CUBES) + " " + uniqueProviders[blockPicker.providerIndex];
                    if (ImGui::BeginCombo("##provider", providerPreview.c_str(), ImGuiComboFlags_NoArrowButton)) {
                        for (int i = 0; i < static_cast<int>(uniqueProviders.size()); ++i) {
                            const bool selected = (i == blockPicker.providerIndex);
                            if (ImGui::Selectable(uniqueProviders[i].c_str(), selected)) {
                                blockPicker.providerIndex = i;
                            }
                            if (selected) ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }

                    ImGui::PopStyleVar(3);
                }

                // Handle keyboard navigation.

                if (ImGui::IsKeyPressed(ImGuiKey_DownArrow, false)) {
                    blockPicker.selectedIndex = (blockPicker.selectedIndex + 1) % ImMax(1, filteredCount);
                    blockPicker.deviceIndex = 0;
                    blockPicker.runtimeIndex = 0;
                    blockPicker.providerIndex = 0;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_UpArrow, false)) {
                    blockPicker.selectedIndex = (blockPicker.selectedIndex - 1 + filteredCount) % ImMax(1, filteredCount);
                    blockPicker.deviceIndex = 0;
                    blockPicker.runtimeIndex = 0;
                    blockPicker.providerIndex = 0;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, false) && !uniqueDevices.empty() && !uniqueRuntimes.empty() && !uniqueProviders.empty()) {
                    blockPicker.providerIndex++;
                    if (blockPicker.providerIndex >= static_cast<int>(uniqueProviders.size())) {
                        blockPicker.providerIndex = 0;
                        blockPicker.runtimeIndex++;
                        if (blockPicker.runtimeIndex >= static_cast<int>(uniqueRuntimes.size())) {
                            blockPicker.runtimeIndex = 0;
                            blockPicker.deviceIndex = (blockPicker.deviceIndex + 1) % static_cast<int>(uniqueDevices.size());
                        }
                    }
                }
                if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, false) && !uniqueDevices.empty() && !uniqueRuntimes.empty() && !uniqueProviders.empty()) {
                    blockPicker.providerIndex--;
                    if (blockPicker.providerIndex < 0) {
                        blockPicker.providerIndex = static_cast<int>(uniqueProviders.size()) - 1;
                        blockPicker.runtimeIndex--;
                        if (blockPicker.runtimeIndex < 0) {
                            blockPicker.runtimeIndex = static_cast<int>(uniqueRuntimes.size()) - 1;
                            blockPicker.deviceIndex--;
                            if (blockPicker.deviceIndex < 0) {
                                blockPicker.deviceIndex = static_cast<int>(uniqueDevices.size()) - 1;
                            }
                        }
                    }
                }

                // Handle enter to create block.

                if ((ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)) &&
                    filteredCount > 0 && !uniqueDevices.empty() && !uniqueRuntimes.empty() && !uniqueProviders.empty()) {
                    const auto& selected = filteredBlocks[blockPicker.selectedIndex];
                    enqueue(MailCreateBlock{
                        flowgraphId,
                        selected.type,
                        blockPicker.gridPosition,
                        uniqueDevices[blockPicker.deviceIndex],
                        uniqueRuntimes[blockPicker.runtimeIndex],
                        uniqueProviders[blockPicker.providerIndex]
                    });
                    blockPicker.active = false;
                }

                // Block list.

                const float listHeight = ImGui::GetContentRegionAvail().y;
                ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f * scalingFactor);
                ImGui::BeginChild("##picker_list", ImVec2(-1, listHeight), false);

                ImDrawList* drawList = ImGui::GetWindowDrawList();
                const float rounding = 8.0f * scalingFactor;
                const float padding = 6.0f * scalingFactor;

                for (int i = 0; i < filteredCount; ++i) {
                    const auto& item = filteredBlocks[i];
                    const bool isSelected = (i == blockPicker.selectedIndex);

                    const float titleHeight = ImGui::GetTextLineHeight();
                    const float summaryFontSize = ImGui::GetFontSize() * 0.85f;
                    const float wrapWidth = ImGui::GetContentRegionAvail().x - padding * 2;
                    const float summaryHeight = ImGui::GetFont()->CalcTextSizeA(summaryFontSize, FLT_MAX, wrapWidth, item.summary.c_str()).y;
                    const float rowHeight = titleHeight + summaryHeight + padding * 2 + 2.0f * scalingFactor;

                    const ImVec2 cellMin = ImGui::GetCursorScreenPos();
                    const ImVec2 cellMax(cellMin.x + ImGui::GetContentRegionAvail().x, cellMin.y + rowHeight);

                    ImGui::PushID(i);
                    ImGui::InvisibleButton("##block_item", ImVec2(-1, rowHeight));
                    const bool hovered = ImGui::IsItemHovered();

                    if (ImGui::IsItemClicked()) {
                        blockPicker.selectedIndex = i;
                    }
                    if (ImGui::IsItemClicked() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) &&
                        !uniqueDevices.empty() && !uniqueRuntimes.empty() && !uniqueProviders.empty()) {
                        enqueue(MailCreateBlock{
                            flowgraphId,
                            item.type,
                            blockPicker.gridPosition,
                            uniqueDevices[blockPicker.deviceIndex],
                            uniqueRuntimes[blockPicker.runtimeIndex],
                            uniqueProviders[blockPicker.providerIndex]
                        });
                        blockPicker.active = false;
                    }

                    // Scroll selected item into view.

                    if (isSelected && (ImGui::IsKeyPressed(ImGuiKey_DownArrow) || ImGui::IsKeyPressed(ImGuiKey_UpArrow))) {
                        ImGui::SetScrollHereY();
                    }

                    const ImU32 bgColor = ImGui::ColorConvertFloat4ToU32(isSelected ? ColorMap.at("header_hovered") : (hovered ? ColorMap.at("cell_background") : ImVec4(0, 0, 0, 0)));
                    drawList->AddRectFilled(cellMin, cellMax, bgColor, rounding);

                    // Title.

                    ImGui::SetCursorScreenPos(ImVec2(cellMin.x + padding, cellMin.y + padding));
                    ImGui::PushFont(boldFont, ImGui::GetFontSize());
                    ImGui::TextUnformatted(item.title.c_str());
                    ImGui::PopFont();

                    // Summary.

                    ImGui::SetCursorScreenPos(ImVec2(cellMin.x + padding, cellMin.y + padding + titleHeight + 2.0f * scalingFactor));
                    ImGui::PushFont(ImGui::GetFont(), summaryFontSize);
                    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + wrapWidth);
                    ImGui::TextWrapped("%s", item.summary.c_str());
                    ImGui::PopTextWrapPos();
                    ImGui::PopStyleColor();
                    ImGui::PopFont();

                    ImGui::PopID();
                }

                if (filteredCount == 0) {
                    const char* noMatchText = "No matching blocks.";
                    const float textWidth = ImGui::CalcTextSize(noMatchText).x;
                    ImGui::SetCursorPosX((ImGui::GetWindowWidth() - textWidth) * 0.5f);
                    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 20.0f * scalingFactor);
                    ImGui::TextDisabled("%s", noMatchText);
                }

                ImGui::EndChild();
                ImGui::PopStyleVar();
            }
            ImGui::End();

            ImGui::PopStyleColor();
            ImGui::PopStyleVar(3);
        }

        // Render runtime metrics below blocks.

        if (debugRuntimeMetricsEnabled) {
            const auto& metricsMap = flowgraph->metrics();

            for (const auto& [blockName, blockPtr] : blocks) {
                if (blockPtr->state() != Block::State::Created) {
                    continue;
                }

                const int nodeId = idFromStr("node:" + flowgraphId + ":" + blockName);
                const ImVec2 nodePos = ImNodes::GetNodeScreenSpacePos(nodeId);
                const ImVec2 nodeSize = ImNodes::GetNodeDimensions(nodeId);

                const std::string moduleName = jst::fmt::format("{}-{}", blockName, blockPtr->config().type());
                if (metricsMap.contains(moduleName)) {
                    const auto& m = metricsMap.at(moduleName);
                    const std::string line1 = jst::fmt::format("Runtime #{} ({}/{})", m->runtime, m->device, m->backend);
                    const std::string line2 = jst::fmt::format("{:.2f} ms | {:.1f}k cycles", m->averageComputeTime, m->cycles / 1000.0f);
                    const ImVec2 textPos(nodePos.x, nodePos.y + nodeSize.y + 4.0f * scalingFactor);
                    const ImU32 color = ImGui::ColorConvertFloat4ToU32(ColorMap.at("text_secondary"));
                    ImDrawList* drawList = ImGui::GetWindowDrawList();
                    drawList->AddText(textPos, color, line1.c_str());
                    drawList->AddText(ImVec2(textPos.x, textPos.y + ImGui::GetTextLineHeight()), color, line2.c_str());
                }
            }
        }

        ImNodes::EndNodeEditor();

        // Block picker handler.

        {
            int hoveredNode = -1;
            const bool isNodeHovered = ImNodes::IsNodeHovered(&hoveredNode);
            const bool isEditorHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);

            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && isEditorHovered && !isNodeHovered) {
                blockPicker.active = true;
                blockPicker.flowgraphId = flowgraphId;
                blockPicker.screenPosition = ImGui::GetMousePos();
                blockPicker.gridPosition = ImNodes::ScreenSpaceToGridSpace(ImGui::GetMousePos());
                blockPicker.searchBuffer[0] = '\0';
                blockPicker.selectedIndex = 0;
                blockPicker.deviceIndex = 0;
                blockPicker.runtimeIndex = 0;
                blockPicker.providerIndex = 0;
            }
        }

        // Handle link creation.

        {
            int startPinId, endPinId;
            if (ImNodes::IsLinkCreated(&startPinId, &endPinId)) {
                auto startIt = pinIdToInfo.find(startPinId);
                auto endIt = pinIdToInfo.find(endPinId);

                if (startIt != pinIdToInfo.end() && endIt != pinIdToInfo.end()) {
                    const auto& input = startIt->second.isInput ? startIt->second : endIt->second;
                    const auto& output = startIt->second.isInput ? endIt->second : startIt->second;

                    enqueue(MailConnectBlock{
                        flowgraphId,
                        input.block,
                        input.port,
                        output.block,
                        output.port
                    });
                }
            }
        }

        // Handle link destruction.

        {
            int linkId;
            if (ImNodes::IsLinkDestroyed(&linkId) && linkIdToConnection.contains(linkId)) {
                const auto& info = linkIdToConnection.at(linkId);
                enqueue(MailDisconnectBlock{flowgraphId, info.consumerName, info.inputSlot});
            }
        }

        // Render tensor metadata tooltip on link hover.

        {
            int linkId;
            if (ImNodes::IsLinkHovered(&linkId) && linkIdToConnection.contains(linkId)) {
                const auto& info = linkIdToConnection.at(linkId);

                if (info.tensorLink->resolved()) {
                    const auto& tensor = info.tensorLink->tensor;

                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(ICON_FA_MEMORY " Tensor Metadata");
                    ImGui::Separator();
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
                    ImGui::TextUnformatted(ICON_FA_CIRCLE_INFO " Click on the end of the link to detach it.");
                    ImGui::PopStyleColor();
                    ImGui::Separator();

                    const auto renderSectionHeader = [this](const char* title,
                                                            const bool withSeparator = true) {
                        if (withSeparator) {
                            ImGui::Separator();
                        }
                        ImGui::PushStyleColor(ImGuiCol_Text, ColorMap.at("text_secondary"));
                        ImGui::TextUnformatted(title);
                        ImGui::PopStyleColor();
                    };

                    const auto renderMetadataTable =
                        [this](const std::string& tableId,
                               const F32 keyColumnWidth,
                               const std::vector<std::pair<const char*, std::string>>& rows) {
                            ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,
                                                ImVec2(6.0f * this->scalingFactor,
                                                       3.0f * this->scalingFactor));

                            if (ImGui::BeginTable(tableId.c_str(),
                                                  2,
                                                  ImGuiTableFlags_SizingFixedFit |
                                                  ImGuiTableFlags_BordersInnerV |
                                                  ImGuiTableFlags_NoPadOuterX)) {
                                ImGui::TableSetupColumn("##meta-key",
                                                        ImGuiTableColumnFlags_WidthFixed,
                                                        keyColumnWidth);
                                ImGui::TableSetupColumn("##meta-value",
                                                        ImGuiTableColumnFlags_WidthStretch);

                                for (const auto& [key, value] : rows) {
                                    ImGui::TableNextRow();
                                    ImGui::TableSetColumnIndex(0);
                                    ImGui::PushStyleColor(ImGuiCol_Text, ColorMap.at("text_secondary"));
                                    ImGui::TextUnformatted(key);
                                    ImGui::PopStyleColor();
                                    ImGui::TableSetColumnIndex(1);
                                    ImGui::TextWrapped("%s", value.c_str());
                                }

                                ImGui::EndTable();
                            }

                            ImGui::PopStyleVar();
                        };

                    renderSectionHeader("Link", false);
                    renderMetadataTable(jst::fmt::format("##tensor-meta-conn-{}", linkId),
                                        72.0f * scalingFactor,
                                        {
                                            {"Path", jst::fmt::format("{}.{} -> {}.{}",
                                                                      info.producerName,
                                                                      info.producerPort,
                                                                      info.consumerName,
                                                                      info.inputSlot)},
                                            {"Tensor ID", jst::fmt::format("{}", tensor.id())},
                                        });

                    renderSectionHeader("Layout");
                    renderMetadataTable(jst::fmt::format("##tensor-meta-layout-{}", linkId),
                                        72.0f * scalingFactor,
                                        {
                                            {"Device", jst::fmt::format("{}", tensor.device())},
                                            {"Type", jst::fmt::format("{}", tensor.dtype())},
                                            {"Shape", jst::fmt::format("{}", tensor.shape())},
                                            {"Strides", jst::fmt::format("{} ({})",
                                                                         tensor.stride(),
                                                                         tensor.contiguous()
                                                                             ? "Contiguous"
                                                                             : "Non-contiguous")},
                                            {"Offset", jst::fmt::format("{} bytes",
                                                                        tensor.offsetBytes())},
                                        });

                    const auto attributeKeys = tensor.attributeKeys();
                    if (!attributeKeys.empty()) {
                        renderSectionHeader("Attributes");

                        const ImGuiTableFlags attributeTableFlags = ImGuiTableFlags_SizingFixedFit |
                                                                    ImGuiTableFlags_BordersInnerV |
                                                                    ImGuiTableFlags_NoPadOuterX;
                        const auto attributeTableId = jst::fmt::format("##tensor-attrs-{}", linkId);

                        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,
                                            ImVec2(6.0f * scalingFactor, 3.0f * scalingFactor));

                        if (ImGui::BeginTable(attributeTableId.c_str(), 2, attributeTableFlags)) {
                            ImGui::TableSetupColumn("##attr-key",
                                                    ImGuiTableColumnFlags_WidthFixed,
                                                    120.0f * scalingFactor);
                            ImGui::TableSetupColumn("##attr-value",
                                                    ImGuiTableColumnFlags_WidthStretch);

                            for (const auto& key : attributeKeys) {
                                std::string encoded;
                                if (Parser::TypedToString(tensor.attribute(key), encoded) != Result::SUCCESS) {
                                    encoded = "?";
                                }

                                ImGui::TableNextRow();
                                ImGui::TableSetColumnIndex(0);
                                ImGui::PushStyleColor(ImGuiCol_Text, ColorMap.at("text_secondary"));
                                ImGui::TextUnformatted(key.c_str());
                                ImGui::PopStyleColor();
                                ImGui::TableSetColumnIndex(1);
                                ImGui::TextWrapped("%s", encoded.c_str());
                            }

                            ImGui::EndTable();
                        }

                        ImGui::PopStyleVar();
                    }

                    ImGui::EndTooltip();
                }
            }
        }

        for (const auto& [nodeId, blockName] : nodeIdToBlockName) {
            NodeMeta nodeMeta;
            flowgraph->getMeta("node", nodeMeta, blockName);

            bool isDetached = false;
            if (blocks.contains(blockName)) {
                const auto& blockPtr = blocks.at(blockName);
                if (blockPtr && !blockPtr->surfaces().empty()) {
                    const auto& surface = blockPtr->surfaces().front();
                    if (!surface->manifests().empty()) {
                        const auto& manifest = surface->manifests().front();
                        SurfaceMeta surfaceMeta;
                        flowgraph->getMeta("surface_" + manifest.id, surfaceMeta, blockName);
                        isDetached = surfaceMeta.detached;
                    }
                }
            }

            const ImVec2 pos = ImNodes::GetNodeGridSpacePos(nodeId);
            const F32 width = nodeSizes[nodeId].x / scalingFactor;
            const F32 height = isDetached ? nodeMeta.height : nodeSizes[nodeId].y / scalingFactor;

            flowgraph->setMeta("node", NodeMeta{pos.x, pos.y, width, height}, blockName);
        }

        static int hoveredNodeId = -1;
        static std::string hoveredFlowgraphId;
        static ImVec2 contextMenuGridPos;

        {
            int currentHoveredNodeId;
            const bool isNodeHovered = ImNodes::IsNodeHovered(&currentHoveredNodeId);
            const bool isEditorHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && isEditorHovered) {
                if (isNodeHovered && nodeIdToBlockName.contains(currentHoveredNodeId)) {
                    hoveredNodeId = currentHoveredNodeId;
                    hoveredFlowgraphId = flowgraphId;
                    ImGui::OpenPopup("##node_context_menu");
                } else if (!isNodeHovered) {
                    hoveredFlowgraphId = flowgraphId;
                    contextMenuGridPos = ImNodes::ScreenSpaceToGridSpace(ImGui::GetMousePos());
                    ImGui::OpenPopup("##flowgraph_context_menu");
                }
            }
        }

        // Keyboard shortcuts for node manipulation.

        {
            ImGuiIO& io = ImGui::GetIO();
            const bool commandPressed = (io.KeyMods & ImGuiMod_Super) != 0;
            const bool controlPressed = (io.KeyMods & ImGuiMod_Ctrl) != 0;

            // Get selected nodes.
            const int numSelectedNodes = ImNodes::NumSelectedNodes();
            std::vector<int> selectedNodes(numSelectedNodes);
            if (numSelectedNodes > 0) {
                ImNodes::GetSelectedNodes(selectedNodes.data());
            }

            // Copy (Ctrl/Cmd+C).
            if ((commandPressed || controlPressed) && ImGui::IsKeyPressed(ImGuiKey_C, false)) {
                if (numSelectedNodes > 0 && nodeIdToBlockName.contains(selectedNodes[0])) {
                    enqueue(MailCopyBlock{flowgraphId, nodeIdToBlockName.at(selectedNodes[0])});
                }
            }

            // Paste (Ctrl/Cmd+V).
            if ((commandPressed || controlPressed) && ImGui::IsKeyPressed(ImGuiKey_V, false)) {
                if (clipboard.hasData) {
                    const ImVec2 gridPos = ImNodes::ScreenSpaceToGridSpace(io.MousePos);
                    enqueue(MailPasteBlock{flowgraphId, gridPos});
                }
            }
        }

        if (hoveredFlowgraphId == flowgraphId && ImGui::BeginPopup("##node_context_menu")) {
            if (ImGui::MenuItem(ICON_FA_COPY " Copy Block", "CTRL+C")) {
                enqueue(MailCopyBlock{flowgraphId, nodeIdToBlockName.at(hoveredNodeId)});
            }
            if (ImGui::MenuItem(ICON_FA_PASTE " Paste Block", "CTRL+V", false, clipboard.hasData)) {
                NodeMeta nodeMeta;
                flowgraph->getMeta("node", nodeMeta, nodeIdToBlockName.at(hoveredNodeId));
                const ImVec2 pastePos = {nodeMeta.x + 50.0f, nodeMeta.y + 50.0f};
                enqueue(MailPasteBlock{flowgraphId, pastePos});
            }
            ImGui::Separator();
            if (ImGui::MenuItem(ICON_FA_ARROW_ROTATE_RIGHT " Reload Block")) {
                enqueue(MailReloadBlock{flowgraphId, nodeIdToBlockName.at(hoveredNodeId)});
            }
            if (ImGui::MenuItem(ICON_FA_XMARK " Delete Block")) {
                enqueue(MailDeleteBlock{flowgraphId, nodeIdToBlockName.at(hoveredNodeId)});
            }
            ImGui::Separator();
            if (ImGui::MenuItem(ICON_FA_BOOK " Documentation")) {
                const std::string docKey = flowgraphId + ":" + nodeIdToBlockName.at(hoveredNodeId);
                openDocumentations[docKey] = true;
            }
            ImGui::EndPopup();
        }

        if (hoveredFlowgraphId == flowgraphId && ImGui::BeginPopup("##flowgraph_context_menu")) {
            if (ImGui::MenuItem(ICON_FA_PASTE " Paste Block", "CTRL+V", false, clipboard.hasData)) {
                enqueue(MailPasteBlock{flowgraphId, contextMenuGridPos});
            }
            ImGui::EndPopup();
        }

        ImGui::PopStyleVar();
        ImGui::PopFont();

        ImGui::End();
    }

    return Result::SUCCESS;
}

//
// Draw notifications.
//

Result DefaultCompositor::renderNotifications() {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, scalingFactor * 12.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ColorMap.at("notification_bg"));
    ImGui::RenderNotifications();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    return Result::SUCCESS;
}

//
// Info HUD.
//

Result DefaultCompositor::renderInfoHud() {
    if (!infoPanelEnabled || fullscreenEnabled) {
        return Result::SUCCESS;
    }

    const ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration |
                                         ImGuiWindowFlags_NoDocking |
                                         ImGuiWindowFlags_AlwaysAutoResize |
                                         ImGuiWindowFlags_NoSavedSettings |
                                         ImGuiWindowFlags_NoFocusOnAppearing |
                                         ImGuiWindowFlags_NoNav |
                                         ImGuiWindowFlags_NoMove |
                                         ImGuiWindowFlags_Tooltip;

    const F32 windowPad = 16.0f * scalingFactor;
    const ImVec2 workPos = ImGui::GetMainViewport()->WorkPos;
    const ImVec2 workSize = ImGui::GetMainViewport()->WorkSize;
    ImVec2 windowPos = {workPos.x + windowPad, workPos.y + workSize.y - windowPad};
    const ImVec2 windowPosPivot = {0.0f, 1.0f};
    ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always, windowPosPivot);
    ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);

    ImGui::SetNextWindowBgAlpha(0.35f);
    ImGui::Begin("Info HUD", nullptr, windowFlags);

    float fps = ImGui::GetIO().Framerate;
    if (fps > 50.0f) {
        ImGui::PushStyleColor(ImGuiCol_Text, ColorMap.at("success_green"));
    }
    ImGui::TextFormatted("{:.0f} Hz", fps);
    if (fps > 50.0f) {
        ImGui::PopStyleColor();
    }
    ImGui::SameLine();
    ImGui::TextFormatted("{}", viewport->name());
    ImGui::TextFormatted("{}", render->info());

    ImGui::End();

    return Result::SUCCESS;
}

//
// Fullscreen HUD.
//

Result DefaultCompositor::renderFullscreenHud() {
    if (!fullscreenEnabled) {
        return Result::SUCCESS;
    }

    const ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration |
                                         ImGuiWindowFlags_NoDocking |
                                         ImGuiWindowFlags_AlwaysAutoResize |
                                         ImGuiWindowFlags_NoSavedSettings |
                                         ImGuiWindowFlags_NoMove |
                                         ImGuiWindowFlags_Tooltip;

    const F32 windowPad = 12.0f * scalingFactor;
    const ImVec2 workPos = ImGui::GetMainViewport()->WorkPos;
    const ImVec2 workSize = ImGui::GetMainViewport()->WorkSize;
    ImVec2 windowPos = {workPos.x + workSize.x - windowPad, windowPad};
    const ImVec2 windowPosPivot = {1.0f, 0.0f};
    ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always, windowPosPivot);
    ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);

    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::Begin("Fullscreen HUD", nullptr, windowFlags);

    ImGui::PushStyleColor(ImGuiCol_Text, ColorMap.at("success_green"));
    ImGui::TextUnformatted(ICON_FA_EXPAND " Fullscreen Mode (Press CTRL+L to exit)");
    ImGui::PopStyleColor();

    ImGui::End();

    return Result::SUCCESS;
}

//
// Welcome HUD.
//

Result DefaultCompositor::renderWelcomeHud() {
    if (!flowgraphs.empty() || ImGui::IsPopupOpen("##help_modal")) {
        return Result::SUCCESS;
    }

    const ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar |
                                         ImGuiWindowFlags_NoResize |
                                         ImGuiWindowFlags_NoCollapse |
                                         ImGuiWindowFlags_NoDocking |
                                         ImGuiWindowFlags_AlwaysAutoResize |
                                         ImGuiWindowFlags_NoSavedSettings |
                                         ImGuiWindowFlags_NoFocusOnAppearing |
                                         ImGuiWindowFlags_NoNav |
                                         ImGuiWindowFlags_NoMove |
                                         ImGuiWindowFlags_NoBackground;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowViewport(vp->ID);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

    ImGui::Begin("Welcome", nullptr, windowFlags);

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImFont* defaultFont = ImGui::GetFont();
    const float baseFontSize = ImGui::GetFontSize();
    const float cardWidth = 560.0f * scalingFactor;
    const float cardSpacing = 8.0f * scalingFactor;

    {
        const float titleFontSize = baseFontSize * 6.0f;
        const char* titleText = "CyberEther";
        ImVec2 titleSize = boldFont->CalcTextSizeA(titleFontSize, FLT_MAX, 0.0f, titleText);

        ImVec2 titlePos = ImGui::GetCursorScreenPos();
        titlePos.x += (cardWidth - titleSize.x) * 0.5f;

        float xOffset = 0.0f;
        const char* p = titleText;
        while (*p) {
            char ch[2] = {*p, '\0'};
            ImVec2 charSize = boldFont->CalcTextSizeA(titleFontSize, FLT_MAX, 0.0f, ch);

            float t = (xOffset + charSize.x * 0.5f) / titleSize.x;
            ImU32 charColor = ImGui::ColorConvertFloat4ToU32(ImVec4(
                (1.0f - t) * 0.231f + t * 0.659f,
                (1.0f - t) * 0.510f + t * 0.333f,
                (1.0f - t) * 0.965f + t * 0.969f,
                1.0f
            ));

            drawList->AddText(boldFont, titleFontSize, ImVec2(titlePos.x + xOffset, titlePos.y), charColor, ch);
            xOffset += charSize.x;
            ++p;
        }

        ImGui::Dummy(ImVec2(0, titleSize.y));

        const float sloganFontSize = baseFontSize * 1.3f;
        const char* slogan = "GPU-accelerated Signal Processing";
        ImVec2 sloganSize = defaultFont->CalcTextSizeA(sloganFontSize, FLT_MAX, 0.0f, slogan);
        ImVec2 sloganPos = ImGui::GetCursorScreenPos();
        sloganPos.x += (cardWidth - sloganSize.x) * 0.5f;
        drawList->AddText(defaultFont, sloganFontSize, sloganPos,
                          ImGui::ColorConvertFloat4ToU32(ColorMap.at("text_secondary")), slogan);
        ImGui::Dummy(ImVec2(0, sloganSize.y + 13.0f * scalingFactor));
    }

    ImGui::Dummy(ImVec2(0, 16.0f * scalingFactor));

    {
        const float actionCardWidth = (cardWidth - cardSpacing * 2.0f) / 3.0f;
        const float actionCardHeight = 120.0f * scalingFactor;
        const float actionCardRadius = 16.0f * scalingFactor;

        const ImU32 cardBgColor = ImGui::ColorConvertFloat4ToU32(ColorMap.at("card"));
        const ImU32 cardBgHoverColor = ImGui::ColorConvertFloat4ToU32(ColorMap.at("header_hovered"));
        const ImU32 borderColor = ImGui::ColorConvertFloat4ToU32(ColorMap.at("border"));
        const ImU32 cyberBlueColor = ImGui::ColorConvertFloat4ToU32(ColorMap.at("cyber_blue"));
        const ImU32 textPrimaryColor = ImGui::ColorConvertFloat4ToU32(ColorMap.at("text_primary"));
        const ImU32 textSecondaryColor = ImGui::ColorConvertFloat4ToU32(ColorMap.at("text_secondary"));

        struct ActionCard {
            const char* icon;
            const char* title;
            const char* subtitle;
            ImU32 iconColor;
            std::function<void()> action;
        };

        ActionCard cards[] = {
            {
                ICON_FA_FILE_CIRCLE_PLUS,
                "New Flowgraph",
                "Start fresh",
                ImGui::ColorConvertFloat4ToU32(ColorMap.at("welcome_icon_new")),
                [this]() { enqueue(MailNewFlowgraph{}); }
            },
            {
                ICON_FA_FOLDER_OPEN,
                "Open File",
                "Load existing",
                ImGui::ColorConvertFloat4ToU32(ColorMap.at("welcome_icon_open")),
                [this]() { helperOpenFlowgraph(); }
            },
            {
                ICON_FA_FLASK,
                "Examples",
                "Quick start",
                ImGui::ColorConvertFloat4ToU32(ColorMap.at("welcome_icon_examples")),
                [this]() { globalModalContent = InterfaceModalContent::FlowgraphExamples; }
            }
        };

        for (int i = 0; i < 3; ++i) {
            const ActionCard& card = cards[i];
            ImVec2 cardPos = ImGui::GetCursorScreenPos();
            ImVec2 cardEnd = ImVec2(cardPos.x + actionCardWidth, cardPos.y + actionCardHeight);
            float cardCenterX = cardPos.x + actionCardWidth * 0.5f;

            ImGui::PushID(i);
            ImGui::InvisibleButton("##action_card", ImVec2(actionCardWidth, actionCardHeight));
            bool hovered = ImGui::IsItemHovered();
            bool clicked = ImGui::IsItemClicked();

            ImU32 cardBg = hovered ? cardBgHoverColor : cardBgColor;
            ImU32 cardBorder = hovered ? cyberBlueColor : borderColor;

            drawList->AddRectFilled(cardPos, cardEnd, cardBg, actionCardRadius);
            drawList->AddRect(cardPos, cardEnd, cardBorder, actionCardRadius, 0, 1.5f * scalingFactor);

            const float iconFontSize = baseFontSize * 2.0f;
            ImVec2 iconSize = defaultFont->CalcTextSizeA(iconFontSize, FLT_MAX, 0.0f, card.icon);
            float iconX = cardCenterX - iconSize.x * 0.5f;
            float iconY = cardPos.y + 20.0f * scalingFactor;
            drawList->AddText(defaultFont, iconFontSize, ImVec2(iconX, iconY), card.iconColor, card.icon);

            const float titleFontSize = baseFontSize * 1.0f;
            ImVec2 titleSize = boldFont->CalcTextSizeA(titleFontSize, FLT_MAX, 0.0f, card.title);
            float titleX = cardCenterX - titleSize.x * 0.5f;
            float titleY = cardPos.y + 65.0f * scalingFactor;
            drawList->AddText(boldFont, titleFontSize, ImVec2(titleX, titleY), textPrimaryColor, card.title);

            const float subtitleFontSize = baseFontSize * 0.9f;
            ImVec2 subtitleSize = defaultFont->CalcTextSizeA(subtitleFontSize, FLT_MAX, 0.0f, card.subtitle);
            float subtitleX = cardCenterX - subtitleSize.x * 0.5f;
            float subtitleY = cardPos.y + 88.0f * scalingFactor;
            drawList->AddText(defaultFont, subtitleFontSize, ImVec2(subtitleX, subtitleY), textSecondaryColor, card.subtitle);

            if (clicked) {
                card.action();
            }

            if (hovered) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            }

            ImGui::PopID();

            if (i < 2) {
                ImGui::SameLine(0.0f, cardSpacing);
            }
        }
    }

    ImGui::Dummy(ImVec2(0, 12.0f * scalingFactor));

    {
        const float shortcutBoxWidth = cardWidth;
        const float shortcutBoxHeight = 55.0f * scalingFactor;
        const float shortcutBoxRadius = 10.0f * scalingFactor;

        const ImU32 panelColor = ImGui::ColorConvertFloat4ToU32(ColorMap.at("panel"));
        const ImU32 borderColor = ImGui::ColorConvertFloat4ToU32(ColorMap.at("border"));
        const ImU32 textSecondaryColor = ImGui::ColorConvertFloat4ToU32(ColorMap.at("text_secondary"));
        const ImU32 textPrimaryColor = ImGui::ColorConvertFloat4ToU32(ColorMap.at("text_primary"));
        const ImU32 buttonColor = ImGui::ColorConvertFloat4ToU32(ColorMap.at("button"));

        ImVec2 boxPos = ImGui::GetCursorScreenPos();
        ImVec2 boxEnd = ImVec2(boxPos.x + shortcutBoxWidth, boxPos.y + shortcutBoxHeight);

        drawList->AddRectFilled(boxPos, boxEnd, panelColor, shortcutBoxRadius);
        drawList->AddRect(boxPos, boxEnd, borderColor, shortcutBoxRadius);

        const char* headerText = ICON_FA_KEYBOARD " Keyboard Shortcuts";
        drawList->AddText(defaultFont, baseFontSize * 0.85f,
                          ImVec2(boxPos.x + 12.0f * scalingFactor, boxPos.y + 8.0f * scalingFactor),
                          textSecondaryColor, headerText);

        struct Shortcut {
            const char* keys;
            const char* action;
        };

        Shortcut shortcuts[] = {
            {"Ctrl+N", "New"},
            {"Ctrl+O", "Open"},
            {"Ctrl+S", "Save"},
            {"Ctrl+T", "Spotlight"}
        };

        float shortcutY = boxPos.y + 30.0f * scalingFactor;
        float shortcutStartX = boxPos.x + 12.0f * scalingFactor;
        float shortcutSpacing = (shortcutBoxWidth - 24.0f * scalingFactor) / 4.0f;

        for (int i = 0; i < 4; ++i) {
            float x = shortcutStartX + i * shortcutSpacing;

            ImVec2 keySize = defaultFont->CalcTextSizeA(baseFontSize * 0.85f, FLT_MAX, 0.0f, shortcuts[i].keys);
            float keyPadX = 5.0f * scalingFactor;
            float keyPadY = 2.0f * scalingFactor;
            ImVec2 keyMin = ImVec2(x, shortcutY);
            ImVec2 keyMax = ImVec2(x + keySize.x + keyPadX * 2.0f, shortcutY + keySize.y + keyPadY * 2.0f);

            drawList->AddRectFilled(keyMin, keyMax, buttonColor, 4.0f * scalingFactor);
            drawList->AddRect(keyMin, keyMax, borderColor, 4.0f * scalingFactor);
            drawList->AddText(defaultFont, baseFontSize * 0.85f,
                              ImVec2(x + keyPadX, shortcutY + keyPadY),
                              textPrimaryColor, shortcuts[i].keys);

            float actionX = keyMax.x + 4.0f * scalingFactor;
            drawList->AddText(defaultFont, baseFontSize * 0.85f,
                              ImVec2(actionX, shortcutY + keyPadY),
                              textSecondaryColor, shortcuts[i].action);
        }

        ImGui::Dummy(ImVec2(shortcutBoxWidth, shortcutBoxHeight));
    }

    ImGui::Dummy(ImVec2(0, 16.0f * scalingFactor));

    {
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(14.0f * scalingFactor, 8.0f * scalingFactor));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8.0f * scalingFactor);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f * scalingFactor, 0.0f));
        ImGui::PushStyleColor(ImGuiCol_Button, ColorMap.at("card"));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ColorMap.at("header_hovered"));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ColorMap.at("header_active"));
        ImGui::PushStyleColor(ImGuiCol_Border, ColorMap.at("border"));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f * scalingFactor);

        const char* websiteBtn = ICON_FA_GLOBE " Website";
        const char* docBtn = ICON_FA_BOOK " Docs";
        const char* repoBtn = ICON_FA_CODE_BRANCH " Repository";
        const char* aboutBtn = ICON_FA_CIRCLE_INFO " About";

        ImVec2 websiteSize = ImGui::CalcTextSize(websiteBtn);
        ImVec2 docSize = ImGui::CalcTextSize(docBtn);
        ImVec2 repoSize = ImGui::CalcTextSize(repoBtn);
        ImVec2 aboutSize = ImGui::CalcTextSize(aboutBtn);

        float framePadX = 14.0f * scalingFactor * 2.0f;
        float btnSpacing = 8.0f * scalingFactor;
        float totalBtnWidth = (websiteSize.x + framePadX) +
                              (docSize.x + framePadX) +
                              (repoSize.x + framePadX) +
                              (aboutSize.x + framePadX) +
                              (btnSpacing * 3.0f);

        ImGui::SetCursorPosX((cardWidth - totalBtnWidth) * 0.5f);

        if (ImGui::Button(websiteBtn)) {
            Platform::OpenUrl("https://cyberether.org");
        }
        ImGui::SameLine();
        if (ImGui::Button(docBtn)) {
            Platform::OpenUrl("https://cyberether.org/docs");
        }
        ImGui::SameLine();
        if (ImGui::Button(repoBtn)) {
            Platform::OpenUrl("https://github.com/luigifcruz/CyberEther");
        }
        ImGui::SameLine();
        if (ImGui::Button(aboutBtn)) {
            globalModalContent = InterfaceModalContent::About;
        }

        ImGui::PopStyleVar(4);
        ImGui::PopStyleColor(4);

        ImGui::Dummy(ImVec2(0, 12.0f * scalingFactor));

        ImGui::PushStyleColor(ImGuiCol_Text, ColorMap.at("text_disabled"));
        std::string versionStr = "v" JETSTREAM_VERSION_STR;
        ImVec2 versionSize = ImGui::CalcTextSize(versionStr.c_str());
        ImGui::SetCursorPosX((cardWidth - versionSize.x) * 0.5f);
        ImGui::TextUnformatted(versionStr.c_str());
        ImGui::PopStyleColor();
    }

    ImGui::End();

    ImGui::PopStyleVar(2);

    return Result::SUCCESS;
}

//
// Remote HUD.
//

Result DefaultCompositor::renderRemoteHud() {
    auto remote = instance->remote();
    if (!remote->started() || fullscreenEnabled) {
        return Result::SUCCESS;
    }

    const ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration |
                                         ImGuiWindowFlags_NoDocking |
                                         ImGuiWindowFlags_AlwaysAutoResize |
                                         ImGuiWindowFlags_NoSavedSettings |
                                         ImGuiWindowFlags_NoFocusOnAppearing |
                                         ImGuiWindowFlags_NoNav |
                                         ImGuiWindowFlags_NoMove |
                                         ImGuiWindowFlags_Tooltip;

    const F32 windowPad = 12.0f * scalingFactor;
    const ImVec2 viewportPos = ImGui::GetMainViewport()->Pos;
    const ImVec2 viewportSize = ImGui::GetMainViewport()->Size;
    ImVec2 windowPos = {viewportPos.x + viewportSize.x - windowPad, viewportPos.y + windowPad};
    const ImVec2 windowPosPivot = {1.0f, 0.0f};
    ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always, windowPosPivot);
    ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);

    const ImVec4& green = ColorMap.at("success_green");
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(green.x, green.y, green.z, 0.08f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(green.x, green.y, green.z, 0.25f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f * scalingFactor, 10.0f * scalingFactor));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::Begin("Remote HUD", nullptr, windowFlags);

    const auto& clients = remote->clients();

    ImGui::PushStyleColor(ImGuiCol_Text, ColorMap.at("success_green"));
    ImGui::TextUnformatted(ICON_FA_TOWER_BROADCAST);
    ImGui::PopStyleColor();
    ImGui::SameLine();

    ImGui::TextFormatted("Remote Sharing ({})", clients.size());

    if (ImGui::IsWindowHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    }
    if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(0)) {
        globalModalContent = InterfaceModalContent::RemoteStreaming;
    }

    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);

    return Result::SUCCESS;
}

//
// Menubar
//

Result DefaultCompositor::renderMenubar() {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    if (ImGui::BeginMainMenuBar()) {
        const float cursorY = ImGui::GetCursorPosY();
        ImGui::SetCursorPosY(cursorY - (ImGui::GetStyle().FontSizeBase * 0.25f));
        ImGui::PushFont(boldFont, ImGui::GetStyle().FontSizeBase * 1.5f);
        ImGui::PushStyleColor(ImGuiCol_Text, ColorMap.at("cyber_blue"));
        if (ImGui::BeginMenu("CyberEther")) {
            ImGui::PopStyleColor();
            ImGui::PopFont();

            if (ImGui::MenuItem("About CyberEther")) {
                globalModalContent = InterfaceModalContent::About;
            }
            if (ImGui::MenuItem("View License")) {
                globalModalContent = InterfaceModalContent::License;
            }
            if (ImGui::MenuItem("Third-Party OSS")) {
                globalModalContent = InterfaceModalContent::ThirdParty;
            }
#ifndef JST_OS_BROWSER
            ImGui::Separator();
            if (ImGui::MenuItem("Quit CyberEther")) {
                std::exit(0);
            }
#endif
            ImGui::EndMenu();
        } else {
            ImGui::PopStyleColor();
            ImGui::PopFont();
        }

        ImGui::SetCursorPosY(cursorY);
        if (ImGui::BeginMenu("Flowgraph")) {
            if (ImGui::MenuItem("New", "CTRL+N", false, true)) {
                enqueue(MailNewFlowgraph{});
            }
            if (ImGui::MenuItem("Open", "CTRL+O", false, true)) {
                helperOpenFlowgraph();
            }
            if (ImGui::MenuItem("Save", "CTRL+S", false, focusedFlowgraph.has_value())) {
                enqueue(MailSaveFlowgraph{focusedFlowgraph.value(), ""});
            }
            if (ImGui::MenuItem("Info", "CTRL+I", false, focusedFlowgraph.has_value())) {
                globalModalContent = InterfaceModalContent::FlowgraphInfo;
            }
            if (ImGui::MenuItem("Close", "CTRL+W", false, focusedFlowgraph.has_value())) {
                enqueue(MailCloseFlowgraph{focusedFlowgraph.value()});
            }
            if (ImGui::MenuItem("Rename", nullptr, false, focusedFlowgraph.has_value())) {
                globalModalContent = InterfaceModalContent::FlowgraphInfo;
            }
            if (ImGui::MenuItem("Reset", nullptr, false, focusedFlowgraph.has_value())) {
                enqueue(MailResetFlowgraph{focusedFlowgraph.value()});
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Open Examples", nullptr, false, true)) {
                globalModalContent = InterfaceModalContent::FlowgraphExamples;
            }
            ImGui::EndMenu();
        }

        ImGui::SetCursorPosY(cursorY);
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Show Info Panel", nullptr, &infoPanelEnabled);
            ImGui::MenuItem("Show Flowgraph", nullptr, &flowgraphEnabled, !flowgraphs.empty());
            ImGui::Separator();
            if (ImGui::MenuItem("Remote Streaming", nullptr, false, instance->remote()->supported())) {
                globalModalContent = InterfaceModalContent::RemoteStreaming;
            }
            ImGui::EndMenu();
        }

        ImGui::SetCursorPosY(cursorY);
        if (ImGui::BeginMenu("Developer")) {
            ImGui::MenuItem("Show Demo Window", nullptr, &debugDemoEnabled);
            ImGui::MenuItem("Show Latency Window", nullptr, &debugLatencyEnabled);
            ImGui::MenuItem("Show Viewport Window", nullptr, &debugViewportEnabled);
            ImGui::MenuItem("Show Runtime Metrics", nullptr, &debugRuntimeMetricsEnabled);
            if (ImGui::MenuItem("Show Benchmarks", nullptr, false, true)) {
                globalModalContent = InterfaceModalContent::Benchmark;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Enable Trace", nullptr, &debugEnableTrace)) {
                if (debugEnableTrace) {
                    JST_LOG_SET_DEBUG_LEVEL(4);
                } else {
                    JST_LOG_SET_DEBUG_LEVEL(JST_LOG_DEBUG_DEFAULT_LEVEL);
                }
            }
            ImGui::EndMenu();
        }

        ImGui::SetCursorPosY(cursorY);
        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("Getting started")) {
                // TODO: Change to the correct getting started guide URL.
                NotifyResultClean(Platform::OpenUrl("https://github.com/luigifcruz/CyberEther"));
            }
            if (ImGui::MenuItem("Luigi's Twitter")) {
                NotifyResultClean(Platform::OpenUrl("https://twitter.com/luigifcruz"));
            }
            if (ImGui::MenuItem("Documentation")) {
                NotifyResultClean(Platform::OpenUrl("https://github.com/luigifcruz/CyberEther"));
            }
            if (ImGui::MenuItem("Open repository")) {
                NotifyResultClean(Platform::OpenUrl("https://github.com/luigifcruz/CyberEther"));
            }
            if (ImGui::MenuItem("Report an issue")) {
                NotifyResultClean(Platform::OpenUrl("https://github.com/luigifcruz/CyberEther/issues"));
            }
            ImGui::Separator();
            if (ImGui::MenuItem("View license")) {
                globalModalContent = InterfaceModalContent::License;
            }
            if (ImGui::MenuItem("Third-Party OSS")) {
                globalModalContent = InterfaceModalContent::ThirdParty;
            }
            ImGui::EndMenu();
        }

        currentHeight += ImGui::GetWindowSize().y;
        ImGui::EndMainMenuBar();
    }

    ImGui::PopStyleVar();

    return Result::SUCCESS;
}

//
// Toolbar
//

Result DefaultCompositor::renderToolbar() {
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetMainViewport()->Pos.x, currentHeight));
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetMainViewport()->Size.x, 0));

    {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(scalingFactor * 16.0f, scalingFactor * 7.0f));
        ImGui::Begin("##ToolBar", nullptr, ImGuiWindowFlags_NoDecoration |
                                           ImGuiWindowFlags_NoNav |
                                           ImGuiWindowFlags_NoDocking |
                                           ImGuiWindowFlags_NoSavedSettings);
        ImGui::PopStyleVar();
        ImGui::PopStyleVar();
        ImGui::PopStyleVar();

        {
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scalingFactor * 12.0f, scalingFactor * 8.0f));

            if (ImGui::Button(ICON_FA_FILE " New")) {
                enqueue(MailNewFlowgraph{});
            }
            ImGui::SameLine();

            if (ImGui::Button(ICON_FA_FOLDER_OPEN " Open")) {
                JST_CHECK(helperOpenFlowgraph());
            }
            ImGui::SameLine();

            if (focusedFlowgraph.has_value()) {
                if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save")) {
                    JST_CHECK(helperSaveFlowgraph(focusedFlowgraph.value()));
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_CIRCLE_XMARK " Close")) {
                    JST_CHECK(helperCloseFlowgraph(focusedFlowgraph.value()));
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_ERASER " Reset")) {
                    enqueue(MailResetFlowgraph{focusedFlowgraph.value()});
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_CIRCLE_INFO " Info")) {
                    globalModalContent = InterfaceModalContent::FlowgraphInfo;
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_LAYER_GROUP " New Stack")) {
                    stacks[jst::fmt::format("Stack {}", stacks.size())] = {true, 0};
                }
                ImGui::SameLine();
            }

#ifdef JST_OS_BROWSER
            ImGui::Dummy(ImVec2(5.0f, 0.0f));
            ImGui::SameLine();

            if (ImGui::Button(ICON_FA_PLUG " Connect WebUSB Device")) {
                if (EM_ASM_INT({ return 'usb' in navigator; }) == 0) {
                    ImGui::InsertNotification({ ImGuiToastType_Error, 10000, "This browser is not compatible with WebUSB. "
                                                                             "Try a Chromium based browser like Chrome, Brave, or Opera GX." });
                } else {
                    EM_ASM({  openUsbDevice(); });
                }
            }
            ImGui::SameLine();
#endif

            ImGui::PopStyleVar();
        }

        currentHeight += ImGui::GetWindowSize().y;
        ImGui::End();
    }

    return Result::SUCCESS;
}

//
// Global Modal
//

Result DefaultCompositor::renderGlobalModal() {
    if (globalModalContent.has_value()) {
        ImGui::OpenPopup("##help_modal");
    }

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("##help_modal", nullptr, ImGuiWindowFlags_AlwaysAutoResize |
                                                        ImGuiWindowFlags_NoTitleBar |
                                                        ImGuiWindowFlags_NoResize |
                                                        ImGuiWindowFlags_NoMove |
                                                        ImGuiWindowFlags_NoScrollbar)) {
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ColorMap.at("modal_input_bg"));

        if (globalModalContent == InterfaceModalContent::About) {
            const ImVec4 textColor = ImGui::GetStyleColorVec4(ImGuiCol_Text);
            const ImVec4 disabledColor = ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);

            auto CenterLine = [&](const char* text, ImFont* font, float sizeFactor, const ImVec4& color) {
                ImFont* useFont = font ? font : ImGui::GetFont();
                ImGui::PushFont(useFont, ImGui::GetFontSize() * sizeFactor);

                const ImVec2 textSize = ImGui::CalcTextSize(text);
                const float available = ImGui::GetContentRegionAvail().x;
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImMax(0.0f, (available - textSize.x) * 0.5f));

                ImGui::PushStyleColor(ImGuiCol_Text, color);
                ImGui::TextUnformatted(text);
                ImGui::PopStyleColor();

                ImGui::PopFont();
            };

            ImGui::Spacing();
            CenterLine("CyberEther", h1Font, 3.0f, textColor);
            CenterLine("The final frontier!", nullptr, 1.15f, disabledColor);
            ImGui::Spacing();
            ImGui::Spacing();
            CenterLine("MIT Licensed", nullptr, 1.0f, textColor);
            CenterLine("Copyright (c) 2021-2025 Luigi F. Cruz", nullptr, 1.0f, textColor);
            CenterLine(jst::fmt::format("v{}-{}", JETSTREAM_VERSION_STR,
                                                  JETSTREAM_BUILD_TYPE).c_str(), nullptr, 1.0f, textColor);
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f * scalingFactor);

            if (ImGui::Button("Close", ImVec2(-1, 40.0f * scalingFactor))) {
                ImGui::CloseCurrentPopup();
            }

            ImGui::PopStyleVar();
        } else if (globalModalContent == InterfaceModalContent::FlowgraphExamples) {
            ImGui::TextUnformatted(ICON_FA_STORE " Flowgraph Examples");
            ImGui::Separator();
            ImGui::Spacing();

            const auto examples = Registry::ListAvailableFlowgraphs();

            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            ImGui::TextWrapped("Pick an example to bootstrap a new flowgraph. "
                               "Examples open in a fresh tab so your work stays intact.");
            ImGui::PopStyleColor();
            ImGui::Spacing();

            if (examples.empty()) {
                ImGui::Text("No example flowgraphs registered.");
            } else {
                const F32 lineHeight = ImGui::GetTextLineHeightWithSpacing();
                const F32 textPadding = lineHeight * 0.35f;
                const F32 rowHeight = (lineHeight * 2.0f) + (textPadding * 2.0f);
                const F32 viewportHeight = ImGui::GetMainViewport()->Size.y;
                const F32 totalTableHeight = rowHeight * std::ceil(examples.size() / 2.0f);
                const F32 minTableHeight = 275.0f * scalingFactor;
                const F32 maxTableHeight = viewportHeight * 0.8f;
                const F32 tableHeight = std::clamp(totalTableHeight, minTableHeight, maxTableHeight);

                const ImGuiTableFlags tableFlags = ImGuiTableFlags_PadOuterX |
                                                   ImGuiTableFlags_NoBordersInBody |
                                                   ImGuiTableFlags_NoBordersInBodyUntilResize |
                                                   ImGuiTableFlags_ScrollY;

                ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f * scalingFactor, 2.0f * scalingFactor));
                if (ImGui::BeginTable("flowgraph_table", 2, tableFlags, ImVec2(0, tableHeight))) {
                    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 0.5f);
                    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 0.5f);

                    U64 cellCount = 0;
                    for (const auto& flowgraph : examples) {
                        if ((cellCount % 2) == 0) {
                            ImGui::TableNextRow();
                        }
                        ImGui::TableSetColumnIndex(cellCount % 2);

                        const ImVec2 cellMin = ImGui::GetCursorScreenPos();
                        const ImVec2 cellSize = ImVec2(ImGui::GetColumnWidth(), rowHeight);
                        const ImVec2 cellMax = ImVec2(cellMin.x + cellSize.x, cellMin.y + cellSize.y);

                        ImGui::InvisibleButton(("cell_button_" + flowgraph.key).c_str(), cellSize);
                        const bool hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNone);

                        ImDrawList* drawList = ImGui::GetWindowDrawList();
                        const float rounding = ImGui::GetStyle().FrameRounding * 1.5f;
                        const ImU32 bgColor = ImGui::ColorConvertFloat4ToU32(hovered ? ColorMap.at("header_hovered")
                                                                                     : ColorMap.at("cell_background"));
                        const ImU32 borderColor = ImGui::ColorConvertFloat4ToU32(ColorMap.at("border"));
                        drawList->AddRectFilled(cellMin, cellMax, bgColor, rounding);
                        drawList->AddRect(cellMin, cellMax, borderColor, rounding);

                        const float wrapWidth = cellSize.x - (textPadding * 2.0f);
                        const float titleHeight = ImGui::CalcTextSize(flowgraph.title.c_str()).y;
                        const float summaryFontSize = ImGui::GetFontSize() * 0.85f;
                        const float summaryHeight = ImGui::GetFont()->CalcTextSizeA(summaryFontSize, wrapWidth, wrapWidth, flowgraph.summary.c_str()).y;
                        const float spacer = lineHeight * 0.35f;
                        const float contentHeight = titleHeight + spacer + summaryHeight;
                        const float startY = cellMin.y + ImMax(textPadding, (rowHeight - contentHeight) * 0.5f);

                        if (ImGui::IsItemClicked()) {
                            std::vector<char> blob(flowgraph.content.begin(), flowgraph.content.end());
                            enqueue(MailOpenFlowgraphBlob{std::move(blob)});
                            ImGui::CloseCurrentPopup();
                        }

                        ImGui::SetCursorScreenPos(ImVec2(cellMin.x + textPadding, startY));
                        ImGui::PushFont(h2Font, ImGui::GetFontSize() * 1.05f);
                        ImGui::TextUnformatted(flowgraph.title.c_str());
                        ImGui::PopFont();
                        ImGui::SameLine();
                        ImGui::TextDisabled(ICON_FA_CIRCLE_QUESTION);
                        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                            ImGui::BeginTooltip();
                            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                            ImGui::TextWrapped("%s", flowgraph.description.c_str());
                            ImGui::PopTextWrapPos();
                            ImGui::EndTooltip();
                        }

                        ImGui::SetCursorScreenPos(ImVec2(cellMin.x + textPadding, startY + titleHeight + spacer));
                        ImGui::PushFont(ImGui::GetFont(), summaryFontSize);
                        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                        ImGui::PushTextWrapPos(cellMax.x - textPadding);
                        ImGui::TextWrapped("%s", flowgraph.summary.c_str());
                        ImGui::PopTextWrapPos();
                        ImGui::PopStyleColor();
                        ImGui::PopFont();

                        cellCount += 1;
                    }

                    ImGui::EndTable();
                }
                ImGui::PopStyleVar();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f * scalingFactor);
            if (ImGui::Button("Close", ImVec2(-1, 40.0f * scalingFactor))) {
                ImGui::CloseCurrentPopup();
            }

            ImGui::PopStyleVar();
        } else if (globalModalContent == InterfaceModalContent::FlowgraphInfo) {
            [&](){
                if (!focusedFlowgraph.has_value()) {
                    ImGui::CloseCurrentPopup();
                    return;
                }

                if (!flowgraphs.contains(focusedFlowgraph.value())) {
                    ImGui::CloseCurrentPopup();
                    return;
                }

                const auto& flowgraph = flowgraphs.at(focusedFlowgraph.value());

                ImGui::TextUnformatted(ICON_FA_CIRCLE_INFO " Flowgraph Information");
                ImGui::Separator();
                ImGui::Spacing();

                bool saveFile = false;
                std::string filenameField = flowgraph->path();

                const ImGuiTableFlags tableFlags = ImGuiTableFlags_PadOuterX;
                if (ImGui::BeginTable("##flowgraph-info-table", 2, tableFlags)) {
                    ImGui::TableSetupColumn("##flowgraph-info-table-labels", ImGuiTableColumnFlags_WidthStretch, 0.20f);
                    ImGui::TableSetupColumn("##flowgraph-info-table-values", ImGuiTableColumnFlags_WidthStretch, 0.80f);

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("Title:");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SetNextItemWidth(-1);
                    auto title = flowgraph->title();
                    if (ImGui::InputText("##flowgraph-info-title", &title)) {
                        JST_CHECK_THROW(flowgraph->setTitle(title));
                    }

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("Summary:");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SetNextItemWidth(-1);
                    auto summary = flowgraph->summary();
                    if (ImGui::InputText("##flowgraph-info-summary", &summary)) {
                        JST_CHECK_THROW(flowgraph->setSummary(summary));
                    }

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("Author:");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SetNextItemWidth(-1);
                    auto author = flowgraph->author();
                    if (ImGui::InputText("##flowgraph-info-author", &author)) {
                        JST_CHECK_THROW(flowgraph->setAuthor(author));
                    }

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("License:");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SetNextItemWidth(-1);
                    auto license = flowgraph->license();
                    if (ImGui::InputText("##flowgraph-info-license", &license)) {
                        JST_CHECK_THROW(flowgraph->setLicense(license));
                    }

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("Description:");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SetNextItemWidth(-1);
                    auto description = flowgraph->description();
                    // TODO: Implement automatic line wrapping.
                    if (ImGui::InputTextMultiline("##flowgraph-info-description", &description)) {
                        JST_CHECK_THROW(flowgraph->setDescription(description));
                    }

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("File Path:");
                    ImGui::TableSetColumnIndex(1);

                    float availableWidth = ImGui::GetContentRegionAvail().x;
                    float buttonWidth = 100.0f * scalingFactor;
                    float inputWidth = availableWidth - buttonWidth - ImGui::GetStyle().ItemSpacing.x;

                    ImGui::SetNextItemWidth(inputWidth);
                    if (ImGui::InputText("##flowgraph-info-filename", &filenameField, ImGuiInputTextFlags_EnterReturnsTrue)) {
                        saveFile |= true;
                    }

                    ImGui::SameLine();
                    if (ImGui::Button("Browse File", ImVec2(buttonWidth, 0))) {
                        const auto& res = Platform::SaveFile(filenameField);
                        if (res == Result::SUCCESS) {
                            saveFile |= true;
                        }
                        NotifyResultClean(res);
                    }

                    ImGui::EndTable();
                }

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();

                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f * scalingFactor);

                ImGui::PushStyleColor(ImGuiCol_Button, ColorMap.at("action_btn"));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ColorMap.at("action_btn_hovered"));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ColorMap.at("action_btn_active"));
                saveFile |= ImGui::Button(ICON_FA_FLOPPY_DISK " Save Flowgraph", ImVec2(-1, 40.0f * scalingFactor));
                ImGui::PopStyleColor(3);
                if (saveFile) {
                    bool validFile = true;
                    if (filenameField.empty()) {
                        JST_ERROR("[FLOWGRAPH] Filename is empty.");
                        NotifyResultClean(Result::ERROR);
                        validFile = false;
                    } else {
                        const std::regex filenamePattern("^.+\\.ya?ml$");
                        if (!std::regex_match(filenameField, filenamePattern)) {
                            JST_ERROR("[FLOWGRAPH] Invalid filename '{}'.", filenameField);
                            NotifyResultClean(Result::ERROR);
                            validFile = false;
                        }
                    }

                    if (validFile) {
                        enqueue(MailSaveFlowgraph{focusedFlowgraph.value(), filenameField});
                        ImGui::CloseCurrentPopup();
                    }
                }

                if (ImGui::Button("Close", ImVec2(-1, 40.0f * scalingFactor))) {
                    ImGui::CloseCurrentPopup();
                }

                ImGui::PopStyleVar();
            }();
        } else if (globalModalContent == InterfaceModalContent::FlowgraphClose) {
            [&](){
                if (!focusedFlowgraph.has_value()) {
                    ImGui::CloseCurrentPopup();
                    return;
                }

                ImGui::TextUnformatted(ICON_FA_TRIANGLE_EXCLAMATION " Close Flowgraph");
                ImGui::Separator();
                ImGui::Spacing();

                ImGui::Text("You are about to close a flowgraph without saving it.");
                ImGui::Text("Are you sure you want to continue?");

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();

                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f * scalingFactor);

                ImGui::PushStyleColor(ImGuiCol_Button, ColorMap.at("action_btn"));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ColorMap.at("action_btn_hovered"));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ColorMap.at("action_btn_active"));
                if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save", ImVec2(-1, 40.0f * scalingFactor))) {
                    globalModalContent = InterfaceModalContent::FlowgraphInfo;
                }
                ImGui::PopStyleColor(3);

                if (ImGui::Button("Don't Save", ImVec2(-1, 40.0f * scalingFactor))) {
                    enqueue(MailCloseFlowgraph{focusedFlowgraph.value()});
                    ImGui::CloseCurrentPopup();
                }

                if (ImGui::Button("Cancel", ImVec2(-1, 40.0f * scalingFactor))) {
                    ImGui::CloseCurrentPopup();
                }

                ImGui::PopStyleVar();
            }();
        } else if (globalModalContent == InterfaceModalContent::RenameBlock) {
            [&](){
                if (!focusedFlowgraph.has_value()) {
                    ImGui::CloseCurrentPopup();
                    return;
                }

                if (!renameBlockOldName.has_value()) {
                    ImGui::CloseCurrentPopup();
                    return;
                }

                ImGui::TextUnformatted(ICON_FA_PENCIL " Rename Block");
                ImGui::Separator();
                ImGui::Spacing();

                std::string renameBlockNewId;

                ImGui::SetNextItemWidth(-1);
                ImGui::InputText("##rename-block-new-id", &renameBlockNewId);

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();

                ImGui::PushStyleColor(ImGuiCol_Text, ColorMap.at("info_blue"));
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f * scalingFactor);

                if (ImGui::Button("Rename Block", ImVec2(-1, 40.0f * scalingFactor))) {
                    enqueue(MailRenameBlock{focusedFlowgraph.value(), renameBlockOldName.value(), renameBlockNewId});
                    ImGui::CloseCurrentPopup();
                }

                ImGui::PopStyleColor();
                if (ImGui::Button("Close", ImVec2(-1, 40.0f * scalingFactor))) {
                    ImGui::CloseCurrentPopup();
                }

                ImGui::PopStyleVar();
            }();
        } else if (globalModalContent == InterfaceModalContent::License) {
            ImGui::TextUnformatted(ICON_FA_KEY " CyberEther License");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("MIT License");

            ImGui::Spacing();

            ImGui::Text("Copyright (c) 2021-2025 Luigi F. Cruz");

            ImGui::Spacing();

            ImGui::Text("Permission is hereby granted, free of charge, to any person obtaining a copy");
            ImGui::Text("of this software and associated documentation files (the \"Software\"), to deal");
            ImGui::Text("in the Software without restriction, including without limitation the rights");
            ImGui::Text("to use, copy, modify, merge, publish, distribute, sublicense, and/or sell");
            ImGui::Text("copies of the Software, and to permit persons to whom the Software is");
            ImGui::Text("furnished to do so, subject to the following conditions:");

            ImGui::Spacing();

            ImGui::Text("The above copyright notice and this permission notice shall be");
            ImGui::Text("included in all copies or substantial portions of the Software.");

            ImGui::Spacing();

            ImGui::Text("THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR");
            ImGui::Text("IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,");
            ImGui::Text("FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE");
            ImGui::Text("AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER");
            ImGui::Text("LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,");
            ImGui::Text("OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE");
            ImGui::Text("SOFTWARE.");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::Button("View Third-Party Licenses", ImVec2(-1, 0))) {
                Platform::OpenUrl("https://cyberether.org/docs/acknowledgments");
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f * scalingFactor);

            if (ImGui::Button("Close", ImVec2(-1, 40.0f * scalingFactor))) {
                ImGui::CloseCurrentPopup();
            }

            ImGui::PopStyleVar();
        } else if (globalModalContent == InterfaceModalContent::ThirdParty) {
            ImGui::TextUnformatted(ICON_FA_BOX_OPEN " Third-Party OSS");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("CyberEther utilizes the following open-source third-party software,");
            ImGui::Text("and we extend our gratitude to the creators of these libraries for");
            ImGui::Text("their valuable contributions to the open-source community.");
            ImGui::Spacing();

            const float listHeight = 240.0f * scalingFactor;
            if (ImGui::BeginChild("oss_scroll", ImVec2(0, listHeight), ImGuiChildFlags_Borders, ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
                ImGui::BulletText("Miniaudio - MIT License");
                ImGui::BulletText("Dear ImGui - MIT License");
                ImGui::BulletText("ImNodes - MIT License");
                ImGui::BulletText("PocketFFT - BSD-3-Clause License");
                ImGui::BulletText("RapidYAML - MIT License");
                ImGui::BulletText("vkFFT - MIT License");
                ImGui::BulletText("stb - MIT License");
                ImGui::BulletText("fmtlib - MIT License");
                ImGui::BulletText("SoapySDR - Boost Software License");
                ImGui::BulletText("libmodes - BSD-2-Clause License");
                ImGui::BulletText("GLFW - zlib/libpng License");
                ImGui::BulletText("imgui-notify - MIT License");
                ImGui::BulletText("spirv-cross - MIT License");
                ImGui::BulletText("glslang - BSD-3-Clause License");
                ImGui::BulletText("naga - Apache License 2.0");
                ImGui::BulletText("gstreamer - LGPL-2.1 License");
                ImGui::BulletText("libusb - LGPL-2.1 License");
                ImGui::BulletText("Nanobench - MIT License");
                ImGui::BulletText("Catch2 - Boost Software License");
                ImGui::BulletText("JetBrains Mono - SIL Open Font License 1.1");
                ImGui::BulletText("imgui_markdown - Zlib License");
                ImGui::BulletText("GLM - Happy Bunny License");
                ImGui::BulletText("cpp-httplib - MIT License");
                ImGui::BulletText("nlohmann/json - MIT License");
                ImGui::BulletText("Natural Earth - Public Domain");
                // [NEW DEPENDENCY HOOK]
            }
            ImGui::EndChild();

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::Button("View Third-Party Licenses", ImVec2(-1, 0))) {
                Platform::OpenUrl("https://cyberether.org/docs/acknowledgments");
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f * scalingFactor);
            if (ImGui::Button("Close", ImVec2(-1, 40.0f * scalingFactor))) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::PopStyleVar();
        } else if (globalModalContent == InterfaceModalContent::Benchmark) {
            ImGui::TextUnformatted(ICON_FA_GAUGE_HIGH " Module Benchmarks");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            ImGui::TextWrapped("Run performance benchmarks for registered modules. "
                               "Results show operations per second and timing information.");
            ImGui::PopStyleColor();
            ImGui::Spacing();

            // Check if benchmark is complete
            if (benchmarkRunning && benchmarkFuture.valid()) {
                if (benchmarkFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    benchmarkRunning = false;
                }
            }

            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f * scalingFactor);

            const float buttonWidth = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;

            ImGui::PushStyleColor(ImGuiCol_Button, ColorMap.at("action_btn"));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ColorMap.at("action_btn_hovered"));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ColorMap.at("action_btn_active"));
            ImGui::BeginDisabled(benchmarkRunning);
            if (ImGui::Button(ICON_FA_PLAY " Run Benchmarks", ImVec2(buttonWidth, 36.0f * scalingFactor))) {
                benchmarkRunning = true;
                benchmarkOutput = std::stringstream();
                Benchmark::ResetResults();

                benchmarkFuture = std::async(std::launch::async, [this]() {
                    Benchmark::Run("quiet", benchmarkOutput);
                });
            }
            ImGui::EndDisabled();
            ImGui::PopStyleColor(3);

            ImGui::SameLine();

            ImGui::BeginDisabled(benchmarkRunning || Benchmark::GetResults().empty());
            if (ImGui::Button(ICON_FA_ROTATE_LEFT " Reset Results", ImVec2(buttonWidth, 36.0f * scalingFactor))) {
                Benchmark::ResetResults();
            }
            ImGui::EndDisabled();

            ImGui::PopStyleVar();

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (benchmarkRunning) {
                const U64 current = Benchmark::CurrentCount();
                const U64 total = Benchmark::TotalCount();
                const float progress = total > 0 ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;

                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ColorMap.at("action_btn"));
                ImGui::ProgressBar(progress, ImVec2(-1, 20.0f * scalingFactor),
                    jst::fmt::format("{:.0f}%", progress * 100.0f).c_str());
                ImGui::PopStyleColor();
                ImGui::Spacing();
            }

            // Show results
            const auto& results = Benchmark::GetResults();

            if (results.empty() && !benchmarkRunning) {
                ImGui::Spacing();
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                const char* emptyText = "No results yet. Click 'Run Benchmarks' to start.";
                const float textWidth = ImGui::CalcTextSize(emptyText).x;
                ImGui::SetCursorPosX((ImGui::GetContentRegionAvail().x - textWidth) * 0.5f + ImGui::GetCursorPosX());
                ImGui::TextUnformatted(emptyText);
                ImGui::PopStyleColor();
                ImGui::Spacing();
            } else if (!results.empty()) {
                const F32 viewportHeight = ImGui::GetMainViewport()->Size.y;
                const F32 minListHeight = 200.0f * scalingFactor;
                const F32 maxListHeight = viewportHeight * 0.5f;
                const F32 listHeight = std::clamp(300.0f * scalingFactor, minListHeight, maxListHeight);

                if (ImGui::BeginChild("benchmark_results", ImVec2(0, listHeight), ImGuiChildFlags_None)) {
                    for (const auto& [module, entries] : results) {
                        ImGui::PushFont(h2Font, ImGui::GetFontSize() * 1.05f);
                        ImGui::TextUnformatted(module.c_str());
                        ImGui::PopFont();
                        ImGui::Spacing();

                        if (ImGui::BeginTable(("BenchmarkResults_" + module).c_str(), 4,
                                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
                            ImGui::TableSetupColumn("Variant");
                            ImGui::TableSetupColumn("Ops/sec");
                            ImGui::TableSetupColumn("ms/Op");
                            ImGui::TableSetupColumn("Error %");
                            ImGui::TableHeadersRow();

                            for (const auto& entry : entries) {
                                ImGui::TableNextRow();
                                ImGui::TableNextColumn();
                                ImGui::TextUnformatted(entry.name.c_str());
                                ImGui::TableNextColumn();
                                ImGui::PushStyleColor(ImGuiCol_Text, ColorMap.at("accent_color"));
                                ImGui::TextFormatted("{:.2f}", entry.opsPerSec);
                                ImGui::PopStyleColor();
                                ImGui::TableNextColumn();
                                ImGui::TextFormatted("{:.4f}", entry.msPerOp);
                                ImGui::TableNextColumn();
                                ImGui::TextFormatted("{:.2f}%", entry.error * 100.0);
                            }

                            ImGui::EndTable();
                        }

                        ImGui::Spacing();
                    }
                }
                ImGui::EndChild();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f * scalingFactor);
            if (ImGui::Button("Close", ImVec2(-1, 40.0f * scalingFactor))) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::PopStyleVar();
        } else if (globalModalContent == InterfaceModalContent::RemoteStreaming) {
            auto remote = instance->remote();
            const bool isStarted = remote->started();

            ImGui::TextUnformatted(ICON_FA_TOWER_BROADCAST " Remote Streaming");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            ImGui::TextWrapped("Stream your session to remote clients via WebRTC. "
                               "Clients can connect by scanning the QR code or using the invite URL.");
            ImGui::PopStyleColor();
            ImGui::Spacing();

            if (!isStarted) {
                ImGui::Spacing();

                // Broker URL Configuration
                ImGui::TextUnformatted("Broker URL");
                ImGui::SameLine();
                ImGui::PushFont(ImGui::GetFont(), ImGui::GetFontSize() * 0.7f);
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 2.0f * scalingFactor);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 4.0f * scalingFactor);
                ImGui::TextDisabled(ICON_FA_CIRCLE_QUESTION);
                ImGui::PopFont();
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted("The WebRTC signaling server URL used to coordinate connections with remote clients.");
                    ImGui::EndTooltip();
                }
                ImGui::SetNextItemWidth(-1);
                char brokerBuf[256];
                strncpy(brokerBuf, remoteBrokerUrl.c_str(), sizeof(brokerBuf) - 1);
                brokerBuf[sizeof(brokerBuf) - 1] = '\0';
                if (ImGui::InputText("##broker_url", brokerBuf, sizeof(brokerBuf))) {
                    remoteBrokerUrl = brokerBuf;
                }
                ImGui::Spacing();

                const char* framerateOptions[] = { "15 FPS", "30 FPS", "60 FPS", "120 FPS" };
                int remoteFramerateIndex = 1;

                if (remoteFramerate == 15) {
                    remoteFramerateIndex = 0;
                } else if (remoteFramerate == 30) {
                    remoteFramerateIndex = 1;
                } else if (remoteFramerate == 60) {
                    remoteFramerateIndex = 2;
                } else if (remoteFramerate == 120) {
                    remoteFramerateIndex = 3;
                }

                if (ImGui::BeginTable("remote_stream_options", 2, ImGuiTableFlags_SizingStretchSame)) {
                    ImGui::TableNextRow();

                    ImGui::TableNextColumn();
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Video Codec");
                    ImGui::SameLine();
                    ImGui::PushFont(ImGui::GetFont(), ImGui::GetFontSize() * 0.7f);
                    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 2.0f * scalingFactor);
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 4.0f * scalingFactor);
                    ImGui::TextDisabled(ICON_FA_CIRCLE_QUESTION);
                    ImGui::PopFont();
                    if (ImGui::IsItemHovered()) {
                        ImGui::BeginTooltip();
                        ImGui::TextUnformatted("The video encoding format used for streaming. Use H264 for universal playback, or AV1 for improved efficiency at the cost of higher CPU usage.");
                        ImGui::EndTooltip();
                    }
                    ImGui::SetNextItemWidth(-1);
                    if (ImGui::BeginCombo("##codec", Jetstream::GetRemoteCodecPrettyName(remoteCodec))) {
                        for (const auto codec : Jetstream::RemoteCodecTypes) {
                            const bool selected = (remoteCodec == codec);

                            if (ImGui::Selectable(Jetstream::GetRemoteCodecPrettyName(codec), selected)) {
                                remoteCodec = codec;
                            }

                            if (selected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }

                    ImGui::TableNextColumn();
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Framerate");
                    ImGui::SameLine();
                    ImGui::PushFont(ImGui::GetFont(), ImGui::GetFontSize() * 0.7f);
                    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 2.0f * scalingFactor);
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 4.0f * scalingFactor);
                    ImGui::TextDisabled(ICON_FA_CIRCLE_QUESTION);
                    ImGui::PopFont();
                    if (ImGui::IsItemHovered()) {
                        ImGui::BeginTooltip();
                        ImGui::TextUnformatted("The number of video frames sent per second. Higher framerates provide smoother video but use more bandwidth.");
                        ImGui::EndTooltip();
                    }
                    ImGui::SetNextItemWidth(-1);
                    if (ImGui::Combo("##framerate", &remoteFramerateIndex, framerateOptions, IM_ARRAYSIZE(framerateOptions))) {
                        if (remoteFramerateIndex == 0) {
                            remoteFramerate = 15;
                        } else if (remoteFramerateIndex == 1) {
                            remoteFramerate = 30;
                        } else if (remoteFramerateIndex == 2) {
                            remoteFramerate = 60;
                        } else if (remoteFramerateIndex == 3) {
                            remoteFramerate = 120;
                        }
                    }

                    ImGui::EndTable();
                }
                ImGui::Spacing();

                if (ImGui::BeginTable("remote_stream_options", 2, ImGuiTableFlags_SizingStretchSame)) {
                    ImGui::TableNextRow();

                    ImGui::TableNextColumn();
                    // Encoder Selection
                    ImGui::AlignTextToFramePadding();
                    ImGui::Text("Encoder");
                    ImGui::SameLine();
                    ImGui::PushFont(ImGui::GetFont(), ImGui::GetFontSize() * 0.7f);
                    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 2.0f * scalingFactor);
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 4.0f * scalingFactor);
                    ImGui::TextDisabled(ICON_FA_CIRCLE_QUESTION);
                    ImGui::PopFont();
                    if (ImGui::IsItemHovered()) {
                        ImGui::BeginTooltip();
                        ImGui::TextUnformatted("Choose how video encoding is handled for the remote stream.");
                        ImGui::EndTooltip();
                    }
                    ImGui::SetNextItemWidth(-1);
                    if (ImGui::BeginCombo("##encoder", Jetstream::GetRemoteEncoderPrettyName(remoteEncoder))) {
                        for (const auto encoder : Jetstream::RemoteEncoderTypes) {
                            const bool selected = (remoteEncoder == encoder);

                            if (ImGui::Selectable(Jetstream::GetRemoteEncoderPrettyName(encoder), selected)) {
                                remoteEncoder = encoder;
                            }

                            if (selected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }

                    ImGui::TableNextColumn();
                    // Client Approval Mode
                    ImGui::AlignTextToFramePadding();
                    ImGui::Text("Client Approval");
                    ImGui::SameLine();
                    ImGui::PushFont(ImGui::GetFont(), ImGui::GetFontSize() * 0.7f);
                    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 2.0f * scalingFactor);
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() - 4.0f * scalingFactor);
                    ImGui::TextDisabled(ICON_FA_CIRCLE_QUESTION);
                    ImGui::PopFont();
                    if (ImGui::IsItemHovered()) {
                        ImGui::BeginTooltip();
                        ImGui::TextUnformatted("Choose whether incoming client connections require manual approval or are approved automatically. Only use auto approval in trusted environments.");
                        ImGui::EndTooltip();
                    }
                    ImGui::SetNextItemWidth(-1);
                    int remoteApprovalIndex = remoteAutoJoinSessions ? 1 : 0;
                    const char* remoteApprovalOptions[] = { "Manual Approval", "Auto Approve" };
                    if (ImGui::Combo("##client_approval", &remoteApprovalIndex, remoteApprovalOptions, IM_ARRAYSIZE(remoteApprovalOptions))) {
                        remoteAutoJoinSessions = (remoteApprovalIndex == 1);
                    }

                    ImGui::EndTable();
                }
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();

                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f * scalingFactor);
                if (ImGui::Button(ICON_FA_PLAY " Start Streaming", ImVec2(-1, 40.0f * scalingFactor))) {
                    Instance::Remote::Config remoteConfig;
                    remoteConfig.broker = remoteBrokerUrl;
                    remoteConfig.codec = remoteCodec;
                    remoteConfig.encoder = remoteEncoder;
                    remoteConfig.autoJoinSessions = remoteAutoJoinSessions;
                    remoteConfig.framerate = remoteFramerate;
                    if (remote->create(remoteConfig) == Result::SUCCESS) {
                        ImGui::InsertNotification({ ImGuiToastType_Success, 5000, "Remote streaming started." });
                    } else {
                        ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "Failed to start remote streaming." });
                    }
                }
                ImGui::PopStyleVar();
            } else {
                if (ImGui::BeginTable("remote_layout", 2, ImGuiTableFlags_None)) {
                    ImGui::TableSetupColumn("QR", ImGuiTableColumnFlags_WidthFixed, 200.0f * scalingFactor);
                    ImGui::TableSetupColumn("Info", ImGuiTableColumnFlags_WidthStretch);

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();

                    const std::string& url = remote->inviteUrl();
                    if (cachedQrUrl != url) {
                        cachedQrUrl = url;
                        cachedQrData.clear();
                        cachedQrWidth = 0;

                        QRcode* qr = QRcode_encodeString8bit(url.c_str(), 0, QR_ECLEVEL_L);
                        if (qr) {
                            cachedQrWidth = qr->width;
                            cachedQrData.resize(cachedQrWidth * cachedQrWidth);
                            for (int i = 0; i < cachedQrWidth * cachedQrWidth; i++) {
                                cachedQrData[i] = qr->data[i] & 1;
                            }
                            QRcode_free(qr);
                        }
                    }

                    if (cachedQrWidth > 0) {
                        const int border = 2;
                        const float moduleSize = 4.0f * scalingFactor;
                        const int totalWidth = cachedQrWidth + border * 2;
                        const float qrSize = totalWidth * moduleSize;

                        ImGui::Dummy(ImVec2(0, 20.0f * scalingFactor));

                        ImVec2 pos = ImGui::GetCursorScreenPos();
                        float columnWidth = ImGui::GetContentRegionAvail().x;
                        pos.x += (columnWidth - qrSize) * 0.5f;

                        ImDrawList* drawList = ImGui::GetWindowDrawList();

                        drawList->AddRectFilled(pos, ImVec2(pos.x + qrSize, pos.y + qrSize), IM_COL32(255, 255, 255, 255));

                        for (int y = 0; y < cachedQrWidth; y++) {
                            for (int x = 0; x < cachedQrWidth; x++) {
                                if (cachedQrData[y * cachedQrWidth + x]) {
                                    ImVec2 p1(pos.x + (x + border) * moduleSize, pos.y + (y + border) * moduleSize);
                                    ImVec2 p2(p1.x + moduleSize, p1.y + moduleSize);
                                    drawList->AddRectFilled(p1, p2, IM_COL32(0, 0, 0, 255));
                                }
                            }
                        }

                        ImGui::Dummy(ImVec2(qrSize, qrSize));

                        if (ImGui::IsItemHovered()) {
                            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                        }
                        if (ImGui::IsItemClicked()) {
                            NotifyResultClean(Platform::OpenUrl(remote->inviteUrl()));
                        }
                    }

                    ImGui::Spacing();

                    const char* openText = ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE " Open in browser";
                    float textWidth = ImGui::CalcTextSize(openText).x;
                    float columnWidth = ImGui::GetContentRegionAvail().x;
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (columnWidth - textWidth) * 0.5f);
                    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    ImGui::TextUnformatted(openText);
                    ImGui::PopStyleColor();
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    }
                    if (ImGui::IsItemClicked()) {
                        NotifyResultClean(Platform::OpenUrl(remote->inviteUrl()));
                    }

                    ImGui::TableNextColumn();

                    ImGui::PushFont(h2Font, ImGui::GetFontSize() * 1.05f);
                    ImGui::TextUnformatted("Connection Info");
                    ImGui::PopFont();
                    ImGui::Spacing();

                    ImGui::TextUnformatted("Room ID:");
                    ImGui::SameLine();
                    ImGui::TextUnformatted(remote->roomId().c_str());
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    }
                    if (ImGui::IsItemClicked()) {
                        ImGui::SetClipboardText(remote->roomId().c_str());
                        ImGui::InsertNotification({ ImGuiToastType_Info, 3000, "Room ID copied to clipboard." });
                    }

                    ImGui::TextUnformatted("Access Token:");
                    ImGui::SameLine();
                    ImGui::TextUnformatted(remote->accessToken().c_str());
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    }
                    if (ImGui::IsItemClicked()) {
                        ImGui::SetClipboardText(remote->accessToken().c_str());
                        ImGui::InsertNotification({ ImGuiToastType_Info, 3000, "Access token copied to clipboard." });
                    }

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::PushFont(h2Font, ImGui::GetFontSize() * 1.05f);
                    ImGui::TextUnformatted("Connected Clients");
                    ImGui::PopFont();
                    ImGui::Spacing();

                    const auto& clients = remote->clients();
                    const auto& waitlist = remote->waitlist();

                    std::set<std::string> pendingIds(waitlist.begin(), waitlist.end());

                    size_t connectedCount = 0;
                    for (const auto& client : clients) {
                        if (!pendingIds.contains(client.sessionId)) {
                            connectedCount++;
                        }
                    }

                    if (connectedCount == 0) {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                        ImGui::TextUnformatted("No connected clients.");
                        ImGui::PopStyleColor();
                    } else {
                        for (const auto& client : clients) {
                            if (pendingIds.contains(client.sessionId)) {
                                continue;
                            }

                            std::string code = client.sessionId.substr(client.sessionId.length() - 6);
                            std::transform(code.begin(), code.end(), code.begin(), ::toupper);

                            ImGui::PushStyleColor(ImGuiCol_Text, ColorMap.at("success_green"));
                            ImGui::TextUnformatted(ICON_FA_CIRCLE_CHECK);
                            ImGui::PopStyleColor();
                            ImGui::SameLine();
                            ImGui::TextUnformatted(code.c_str());
                        }
                    }

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::PushFont(h2Font, ImGui::GetFontSize() * 1.05f);
                    ImGui::TextUnformatted("Pending Connections");
                    ImGui::PopFont();
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    ImGui::TextUnformatted("(Click to approve)");
                    ImGui::PopStyleColor();
                    ImGui::Spacing();

                    if (waitlist.empty()) {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                        ImGui::TextUnformatted("No pending connections.");
                        ImGui::PopStyleColor();
                    } else {
                        for (const auto& sessionId : waitlist) {
                            std::string code = sessionId.substr(sessionId.length() - 6);
                            std::transform(code.begin(), code.end(), code.begin(), ::toupper);

                            ImGui::PushID(sessionId.c_str());
                            ImGui::PushStyleColor(ImGuiCol_Text, ColorMap.at("warning_yellow"));
                            ImGui::TextUnformatted(ICON_FA_CLOCK);
                            ImGui::PopStyleColor();
                            ImGui::SameLine();
                            ImGui::TextUnformatted(code.c_str());
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                            }
                            if (ImGui::IsItemClicked()) {
                                if (remote->approveClient(code) == Result::SUCCESS) {
                                    ImGui::InsertNotification({ ImGuiToastType_Success, 3000, "Client approved." });
                                } else {
                                    ImGui::InsertNotification({ ImGuiToastType_Error, 3000, "Failed to approve client." });
                                }
                            }
                            ImGui::PopID();
                        }
                    }

                    ImGui::EndTable();
                }

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();

                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f * scalingFactor);
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.2f, 0.2f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.7f, 0.3f, 0.3f, 1.0f));
                if (ImGui::Button(ICON_FA_STOP " Stop Streaming", ImVec2(-1, 40.0f * scalingFactor))) {
                    if (remote->destroy() == Result::SUCCESS) {
                        ImGui::InsertNotification({ ImGuiToastType_Info, 5000, "Remote streaming stopped." });
                    }
                }
                ImGui::PopStyleColor(2);
                ImGui::PopStyleVar();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f * scalingFactor);
            if (ImGui::Button("Close", ImVec2(-1, 40.0f * scalingFactor))) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::PopStyleVar();
        }

        ImGui::PopStyleColor();
        ImGui::SetItemDefaultFocus();
        ImGui::Dummy(ImVec2(550.0f * scalingFactor, 0.0f));
        ImGui::EndPopup();
    }

    if (!ImGui::IsPopupOpen("##help_modal")) {
        globalModalContent.reset();
    }

    return Result::SUCCESS;
}

//
// Helpers
//

Result DefaultCompositor::helperOpenFlowgraph() {
    std::string path;
    Platform::PickFile(path, {"yaml", "yml"}, [this](std::string p) {
        if (!std::filesystem::exists(p)) {
            ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "The selected file does not exist." });
            return;
        }
        enqueue(MailOpenFlowgraphPath{std::move(p)});
    });

    return Result::SUCCESS;
}

Result DefaultCompositor::helperSaveFlowgraph(const std::string& flowgraph) {
    if (!flowgraphs.contains(flowgraph)) {
        ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "Cannot save flowgraph because it doesn't exist." });
        return Result::SUCCESS;
    }
    std::string path = flowgraphs.at(flowgraph)->path();

    if (!path.empty()) {
        enqueue(MailSaveFlowgraph{flowgraph, std::move(path)});
    } else {
        std::string pickedPath;
        Platform::SaveFile(pickedPath, [this, flowgraph](std::string p) {
            enqueue(MailSaveFlowgraph{flowgraph, std::move(p)});
        });
    }

    return Result::SUCCESS;
}

Result DefaultCompositor::helperCloseFlowgraph(const std::string& flowgraph) {
    if (!flowgraphs.contains(flowgraph)) {
        ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "Cannot close flowgraph because it doesn't exist." });
        return Result::SUCCESS;
    }

    if (flowgraphs.at(flowgraph)->path().empty()) {
        globalModalContent = InterfaceModalContent::FlowgraphClose;
        return Result::SUCCESS;
    }

    enqueue(MailCloseFlowgraph{flowgraph});

    return Result::SUCCESS;
}

void DefaultCompositor::helperRenderLoadingBar(const ImVec4& color, F32 height) {
    const float width = ImGui::GetContentRegionAvail().x;
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    const float time = static_cast<float>(ImGui::GetTime());
    const float t = (sinf(time * 3.0f) + 1.0f) * 0.5f;

    const float glowWidth = width * 0.4f;
    const float centerX = pos.x + t * width;

    for (float x = pos.x; x < pos.x + width; x += 1.0f) {
        const float dist = fabsf(x - centerX);
        const float alpha = fmaxf(0.0f, 1.0f - (dist / (glowWidth * 0.5f)));
        const float glow = alpha * alpha * alpha;

        if (glow > 0.01f) {
            drawList->AddLine(
                ImVec2(x, pos.y),
                ImVec2(x, pos.y + height),
                IM_COL32(
                    static_cast<int>(color.x * 255),
                    static_cast<int>(color.y * 255),
                    static_cast<int>(color.z * 255),
                    static_cast<int>(glow * 255)));
        }
    }

    ImGui::Dummy(ImVec2(width, height));
}

void DefaultCompositor::helperSurfaceResize(const std::shared_ptr<Module::Surface>& surface,
                                            U64 width,
                                            U64 height,
                                            F32 scalingFactor,
                                            bool detached) {
    const auto& bg = ColorMap.at(detached ? "background" : "node_background");
    SurfaceEvent event;
    event.type = SurfaceEventType::Resize;
    event.size = {width, height};
    event.scale = scalingFactor;
    event.backgroundColor = {bg.x, bg.y, bg.z, bg.w};
    surface->pushSurfaceEvent(event);
}

bool DefaultCompositor::helperCheckSurfaceResize(const std::shared_ptr<Module::Surface>& surface,
                                                 const SurfaceManifest& manifest,
                                                 const ImVec2& availableRegion,
                                                 U64& storedWidth,
                                                 U64& storedHeight,
                                                 F32 scalingFactor,
                                                 bool detached) {
    if (availableRegion.x <= 0.0f || availableRegion.y <= 0.0f) {
        return false;
    }

    const U64 newWidth = static_cast<U64>(availableRegion.x / scalingFactor);
    const U64 newHeight = static_cast<U64>(availableRegion.y / scalingFactor);
    const U64 expectedWidth = static_cast<U64>(availableRegion.x * ImGui::GetIO().DisplayFramebufferScale.x);
    const U64 expectedHeight = static_cast<U64>(availableRegion.y * ImGui::GetIO().DisplayFramebufferScale.y);

    if (newWidth == 0 || newHeight == 0 || expectedWidth == 0 || expectedHeight == 0) {
        return false;
    }

    if (storedWidth != newWidth || storedHeight != newHeight ||
        manifest.size.x != expectedWidth || manifest.size.y != expectedHeight) {
        storedWidth = newWidth;
        storedHeight = newHeight;
        helperSurfaceResize(surface, expectedWidth, expectedHeight, scalingFactor, detached);
        return true;
    }
    return false;
}

bool DefaultCompositor::helperRenderSurfaceContent(const std::shared_ptr<Module::Surface>& surface,
                                                   const SurfaceManifest& manifest,
                                                   const ImVec2& surfaceSize,
                                                   float rounding,
                                                   bool forwardMouseEvents) {
    if (surfaceSize.x <= 0.0f || surfaceSize.y <= 0.0f) {
        return false;
    }

    const ImVec2 cursorPos = ImGui::GetCursorScreenPos();
    const ImVec2 cursorEnd = ImVec2(cursorPos.x + surfaceSize.x, cursorPos.y + surfaceSize.y);

    ImGui::GetWindowDrawList()->AddImageRounded(
        ImTextureRef(manifest.surface->raw()),
        cursorPos,
        cursorEnd,
        ImVec2(0, 0),
        ImVec2(1, 1),
        IM_COL32_WHITE,
        rounding);

    ImGui::InvisibleButton(jst::fmt::format("##surface_{}", manifest.id).c_str(), surfaceSize);
    const bool hovered = ImGui::IsItemHovered();

    if (forwardMouseEvents && hovered) {
        const ImVec2 mousePos = ImGui::GetMousePos();
        const Extent2D<F32> normalizedPos = {
            (mousePos.x - cursorPos.x) / surfaceSize.x,
            (mousePos.y - cursorPos.y) / surfaceSize.y
        };

        MouseEvent event;
        event.position = normalizedPos;
        event.scroll = {0.0f, 0.0f};

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            event.type = MouseEventType::Click;
            event.button = MouseButton::Left;
            surface->pushMouseEvent(event);
        } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            event.type = MouseEventType::Click;
            event.button = MouseButton::Right;
            surface->pushMouseEvent(event);
        } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            event.type = MouseEventType::Release;
            event.button = MouseButton::Left;
            surface->pushMouseEvent(event);
        } else if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
            event.type = MouseEventType::Release;
            event.button = MouseButton::Right;
            surface->pushMouseEvent(event);
        }

        const ImGuiIO& io = ImGui::GetIO();
        if (io.MouseWheel != 0.0f || io.MouseWheelH != 0.0f) {
            event.type = MouseEventType::Scroll;
            event.scroll = {io.MouseWheelH, io.MouseWheel};
            surface->pushMouseEvent(event);
        }

        event.type = MouseEventType::Move;
        surface->pushMouseEvent(event);
    }

    return hovered;
}

//
// ImGui Methods
//

void DefaultCompositor::ImGuiLoadFonts() {
    const auto& scalingFactor = render->scalingFactor();
    auto& io = ImGui::GetIO();

    ImFontConfig font_config;
    font_config.OversampleH = 5;
    font_config.OversampleV = 5;
    font_config.FontLoaderFlags = 1;
    io.Fonts->Clear();

    bodyFont = io.Fonts->AddFontFromMemoryCompressedTTF(jbmm_compressed_data,
                                                        jbmm_compressed_size,
                                                        15.0f * scalingFactor,
                                                        &font_config,
                                                        nullptr);

    ImFontConfig iconFontConfig;
    iconFontConfig.OversampleH = 5;
    iconFontConfig.OversampleV = 5;
    iconFontConfig.FontLoaderFlags = 1;
    iconFontConfig.MergeMode = true;
    iconFontConfig.GlyphMinAdvanceX = 15.0f * scalingFactor;
    iconFontConfig.GlyphOffset = { 0.0f, 2.0f };

    static const ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };

    io.Fonts->AddFontFromMemoryCompressedTTF(far_compressed_data,
                                             far_compressed_size,
                                             15.0f * scalingFactor,
                                             &iconFontConfig,
                                             icon_ranges);

    io.Fonts->AddFontFromMemoryCompressedTTF(fas_compressed_data,
                                             fas_compressed_size,
                                             15.0f * scalingFactor,
                                             &iconFontConfig,
                                             icon_ranges);

    h1Font = io.Fonts->AddFontFromMemoryCompressedTTF(jbmb_compressed_data,
                                                      jbmb_compressed_size,
                                                      15.0f * scalingFactor * 1.15,
                                                      &font_config,
                                                      nullptr);

    h2Font = io.Fonts->AddFontFromMemoryCompressedTTF(jbmb_compressed_data,
                                                      jbmb_compressed_size,
                                                      15.0f * scalingFactor * 1.10,
                                                      &font_config,
                                                      nullptr);

    boldFont = io.Fonts->AddFontFromMemoryCompressedTTF(jbmb_compressed_data,
                                                        jbmb_compressed_size,
                                                        15.0f * scalingFactor * 1.04,
                                                        &font_config,
                                                        nullptr);
}

void DefaultCompositor::ImGuiStyleSetup() {
    auto& io = ImGui::GetIO();
    io.ConfigWindowsMoveFromTitleBarOnly = true;

    auto& colors = ImGui::GetStyle().Colors;

    // Text
    colors[ImGuiCol_Text]                      = ColorMap.at("text_primary");
    colors[ImGuiCol_TextDisabled]              = ColorMap.at("text_secondary");
    colors[ImGuiCol_TextSelectedBg]            = ColorMap.at("text_selected_bg");

    // Backgrounds
    colors[ImGuiCol_WindowBg]                  = ColorMap.at("background");
    colors[ImGuiCol_PopupBg]                   = ColorMap.at("popup_bg");
    colors[ImGuiCol_ModalWindowDimBg]          = ColorMap.at("modal_dim");

    // Borders
    colors[ImGuiCol_Border]                    = ColorMap.at("border");
    colors[ImGuiCol_BorderShadow]              = ColorMap.at("border_shadow");

    // Frames (inputs, etc)
    colors[ImGuiCol_FrameBg]                   = ColorMap.at("card");
    colors[ImGuiCol_FrameBgHovered]            = ColorMap.at("frame_bg_hovered");
    colors[ImGuiCol_FrameBgActive]             = ColorMap.at("frame_bg_active");

    // Title bars
    colors[ImGuiCol_TitleBg]                   = ColorMap.at("panel");
    colors[ImGuiCol_TitleBgActive]             = ColorMap.at("title_bg_active");
    colors[ImGuiCol_TitleBgCollapsed]          = ColorMap.at("title_bg_collapsed");

    // Menu bar
    colors[ImGuiCol_MenuBarBg]                 = ColorMap.at("background");

    // Scrollbar
    colors[ImGuiCol_ScrollbarBg]               = ColorMap.at("scrollbar_bg");
    colors[ImGuiCol_ScrollbarGrab]             = ColorMap.at("scrollbar_grab");
    colors[ImGuiCol_ScrollbarGrabHovered]      = ColorMap.at("scrollbar_grab_hovered");
    colors[ImGuiCol_ScrollbarGrabActive]       = ColorMap.at("scrollbar_grab_active");

    // Checkmark & Sliders
    colors[ImGuiCol_CheckMark]                 = ColorMap.at("accent_color");
    colors[ImGuiCol_SliderGrab]                = ColorMap.at("accent_color");
    colors[ImGuiCol_SliderGrabActive]          = ColorMap.at("accent_active");

    // Buttons
    colors[ImGuiCol_Button]                    = ColorMap.at("button");
    colors[ImGuiCol_ButtonHovered]             = ColorMap.at("button_hovered");
    colors[ImGuiCol_ButtonActive]              = ColorMap.at("button_active");

    // Headers
    colors[ImGuiCol_Header]                    = ColorMap.at("header");
    colors[ImGuiCol_HeaderHovered]             = ColorMap.at("header_hovered");
    colors[ImGuiCol_HeaderActive]              = ColorMap.at("header_active");

    // Separators
    colors[ImGuiCol_Separator]                 = ColorMap.at("separator");
    colors[ImGuiCol_SeparatorHovered]          = ColorMap.at("separator_hovered");
    colors[ImGuiCol_SeparatorActive]           = ColorMap.at("separator_active");

    // Resize grip
    colors[ImGuiCol_ResizeGrip]                = ColorMap.at("resize_grip");
    colors[ImGuiCol_ResizeGripHovered]         = ColorMap.at("resize_grip_hovered");
    colors[ImGuiCol_ResizeGripActive]          = ColorMap.at("resize_grip_active");

    // Tabs
    colors[ImGuiCol_Tab]                       = ColorMap.at("tab");
    colors[ImGuiCol_TabHovered]                = ColorMap.at("tab_hovered");
    colors[ImGuiCol_TabSelected]               = ColorMap.at("tab_selected");
    colors[ImGuiCol_TabDimmed]                 = ColorMap.at("tab_dimmed");
    colors[ImGuiCol_TabDimmedSelected]         = ColorMap.at("tab_dimmed_selected");
    colors[ImGuiCol_TabDimmedSelectedOverline] = ColorMap.at("tab_dimmed_selected");
    colors[ImGuiCol_TabSelectedOverline]       = ColorMap.at("tab_selected");

    // Docking
    colors[ImGuiCol_DockingPreview]            = ColorMap.at("docking_preview");
    colors[ImGuiCol_DockingEmptyBg]            = ColorMap.at("docking_empty_bg");

    // Plots
    colors[ImGuiCol_PlotLines]                 = ColorMap.at("plot_lines");
    colors[ImGuiCol_PlotLinesHovered]          = ColorMap.at("plot_lines_hovered");
    colors[ImGuiCol_PlotHistogram]             = ColorMap.at("plot_histogram");
    colors[ImGuiCol_PlotHistogramHovered]      = ColorMap.at("plot_histogram_hovered");

    // Tables
    colors[ImGuiCol_TableHeaderBg]             = ColorMap.at("table_header_bg");
    colors[ImGuiCol_TableBorderStrong]         = ColorMap.at("table_border_strong");
    colors[ImGuiCol_TableBorderLight]          = ColorMap.at("table_border_light");
    colors[ImGuiCol_TableRowBg]                = ColorMap.at("table_row_bg");
    colors[ImGuiCol_TableRowBgAlt]             = ColorMap.at("table_row_bg_alt");

    // Drag & Drop
    colors[ImGuiCol_DragDropTarget]            = ColorMap.at("drag_drop_target");

    // Navigation
    colors[ImGuiCol_NavCursor]                 = ColorMap.at("accent_color");
    colors[ImGuiCol_NavWindowingHighlight]     = ColorMap.at("nav_windowing_highlight");
    colors[ImGuiCol_NavWindowingDimBg]         = ColorMap.at("nav_windowing_dim_bg");
}

void DefaultCompositor::ImGuiStyleScale() {
    auto& style = ImGui::GetStyle();

    // Main
    style.WindowPadding                     = ImVec2(12.00f, 12.00f);
    style.FramePadding                      = ImVec2(12.00f, 4.00f);
    style.ItemSpacing                       = ImVec2(8.00f, 8.00f);
    style.ItemInnerSpacing                  = ImVec2(8.00f, 6.00f);
    style.TouchExtraPadding                 = ImVec2(0.00f, 0.00f);
    style.CellPadding                       = ImVec2(6.00f, 4.00f);
    style.IndentSpacing                     = 20;
    style.ScrollbarSize                     = 12;
    style.GrabMinSize                       = 12;

    // Borders
    style.WindowBorderSize                  = 0.5f;
    style.ChildBorderSize                   = 0.5f;
    style.PopupBorderSize                   = 0.5f;
    style.FrameBorderSize                   = 0.0f;
    style.TabBorderSize                     = 0.0f;
    style.TabBarBorderSize                  = 1.0f;

    // Rounding
    style.WindowRounding                    = 18.0f;
    style.ChildRounding                     = 0.0f;
    style.FrameRounding                     = 6.0f;
    style.PopupRounding                     = 18.0f;
    style.ScrollbarRounding                 = 18.0f;
    style.GrabRounding                      = 8.0f;
    style.LogSliderDeadzone                 = 4.0f;
    style.TabRounding                       = 10.0f;

    // Alignment
    style.WindowTitleAlign                  = ImVec2(0.5f, 0.5f);

    // Tessellation
    style.CircleTessellationMaxError        = 0.1f;
}

//
// ImNodes Methods
//

void DefaultCompositor::ImNodesStyleSetup() {
    auto& colors = ImNodes::GetStyle().Colors;
    colors[ImNodesCol_NodeBackground]         = ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_background"));
    colors[ImNodesCol_NodeBackgroundHovered]  = ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_background"));
    colors[ImNodesCol_NodeBackgroundSelected] = ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_background"));
    colors[ImNodesCol_NodeOutline]            = ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_outline"));
    colors[ImNodesCol_TitleBar]               = ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_title_bar"));
    colors[ImNodesCol_TitleBarHovered]        = ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_title_bar"));
    colors[ImNodesCol_TitleBarSelected]       = ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_title_bar"));
    colors[ImNodesCol_Pin]                    = ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_pin"));
    colors[ImNodesCol_PinHovered]             = ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_pin"));
    colors[ImNodesCol_Link]                   = ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_link"));
    colors[ImNodesCol_LinkHovered]            = ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_link"));
    colors[ImNodesCol_LinkSelected]           = ImGui::ColorConvertFloat4ToU32(ColorMap.at("node_link"));
    colors[ImNodesCol_GridLine]               = ImGui::ColorConvertFloat4ToU32(ColorMap.at("grid_line"));
    colors[ImNodesCol_GridBackground]         = ImGui::ColorConvertFloat4ToU32(ColorMap.at("grid_background"));
}

void DefaultCompositor::ImNodesStyleScale() {
    const auto& scalingFactor = render->scalingFactor();
    auto& style = ImNodes::GetStyle();
    style.NodePadding               = ImVec2(6.0f * scalingFactor,  6.0f * scalingFactor);
    style.PinCircleRadius           = 4.0f  * scalingFactor;
    style.GridSpacing               = 23.0f * scalingFactor;
    style.NodeBorderThickness       = 2.0f  * scalingFactor;
    style.NodeCornerRounding        = 12.0f * scalingFactor;
    style.LinkThickness             = 1.5f  * scalingFactor;
    style.PinLineThickness          = 1.0f  * scalingFactor;
    style.LinkLineSegmentsPerLength = 0.2f  / scalingFactor;
    style.MiniMapOffset             = ImVec2(8.0f * scalingFactor, 8.0f * scalingFactor);
}

//
// ImGui Markdown Methods
//

void DefaultCompositor::ImGuiMarkdownStyleSetup() {
    markdownConfig.linkCallback        = &DefaultCompositor::ImGuiMarkdownLinkCallback;
    markdownConfig.tooltipCallback     = nullptr;
    markdownConfig.imageCallback       = nullptr;
    markdownConfig.linkIcon            = ICON_FA_LINK;
    markdownConfig.headingFormats[0]   = { h1Font, true };
    markdownConfig.headingFormats[1]   = { h2Font, true };
    markdownConfig.headingFormats[2]   = { boldFont, false };
    markdownConfig.userData            = this;
    markdownConfig.formatCallback      = &DefaultCompositor::ImGuiMarkdownFormatCallback;
}

void DefaultCompositor::ImGuiMarkdownLinkCallback(ImGui::MarkdownLinkCallbackData data) {
    if (!data.isImage) {
        std::string url(data.link, data.linkLength);
        Platform::OpenUrl(url);
    }
}

void DefaultCompositor::ImGuiMarkdownFormatCallback(const ImGui::MarkdownFormatInfo& mdInfo, bool start) {
    switch (mdInfo.type) {
        case ImGui::MarkdownFormatType::NORMAL_TEXT:
            break;
        case ImGui::MarkdownFormatType::EMPHASIS: {
            ImGui::MarkdownHeadingFormat fmt;
            fmt = mdInfo.config->headingFormats[ImGui::MarkdownConfig::NUMHEADINGS - 1];
            if (start) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                if (fmt.font) {
                    ImGui::PushFont(fmt.font, 0.0f);
                }
            } else {
                if (fmt.font) {
                    ImGui::PopFont();
                }
                ImGui::PopStyleColor();
            }
            break;
        }
        case ImGui::MarkdownFormatType::HEADING: {
            ImGui::MarkdownHeadingFormat fmt;
            if (mdInfo.level > ImGui::MarkdownConfig::NUMHEADINGS) {
                fmt = mdInfo.config->headingFormats[ImGui::MarkdownConfig::NUMHEADINGS - 1];
            } else {
                fmt = mdInfo.config->headingFormats[mdInfo.level - 1];
            }
            if (start) {
                if (fmt.font) {
                    ImGui::PushFont(fmt.font, 0.0f);
                }
            } else {
                if (fmt.separator) {
                    ImGui::Separator();
                }
                if (fmt.font) {
                    ImGui::PopFont();
                }
            }
            break;
        }
        case ImGui::MarkdownFormatType::UNORDERED_LIST:
            break;
        case ImGui::MarkdownFormatType::LINK: {
            if (start) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertFloat4ToU32(ImVec4(0.278f, 0.498f, 0.937f, 1.0f)));
            } else {
                ImGui::PopStyleColor();
                if (mdInfo.itemHovered) {
                    ImGui::UnderLine(ImGui::ColorConvertFloat4ToU32(ImVec4(0.278f, 0.498f, 0.937f, 1.0f)));
                }
            }
            break;
        }
    }
}

Result DefaultCompositor::renderBackground() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const ImVec4 backgroundColor = ColorMap.at("background");

    const ImVec2 rectMin = ImVec2(viewport->Pos.x, viewport->Pos.y);
    const ImVec2 rectMax = ImVec2(viewport->Pos.x + viewport->Size.x, viewport->Pos.y + viewport->Size.y);

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    drawList->AddRectFilled(rectMin, rectMax, ImGui::ColorConvertFloat4ToU32(backgroundColor));

    return Result::SUCCESS;
}

Result DefaultCompositor::renderDebugLatency() {
    if (!debugLatencyEnabled) {
        return Result::SUCCESS;
    }

    const ImGuiIO& io = ImGui::GetIO();
    const F32 mainWindowWidth = io.DisplaySize.x;
    const F32 mainWindowHeight = io.DisplaySize.y;

    const F32 timerWindowWidth = 200.0f * scalingFactor;
    const F32 timerWindowHeight = 120.0f * scalingFactor;

    static F32 x = 0.0f;
    static F32 xd = 1.0f;

    x += xd;

    if (x > (mainWindowWidth - timerWindowWidth)) {
        xd = -xd;
    }
    if (x < 0.0f) {
        xd = -xd;
    }

    ImGui::SetNextWindowSize(ImVec2(timerWindowWidth, timerWindowHeight));
    ImGui::SetNextWindowPos(ImVec2(x, (mainWindowHeight * 0.25f) - (timerWindowHeight * 0.5f)));

    if (!ImGui::Begin("Latency Debug", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return Result::SUCCESS;
    }

    const U64 ms = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    ImGui::TextFormatted("Time: {} ms", ms);

    const F32 blockWidth = timerWindowWidth / 7.0f;
    const F32 blockHeight = 25.0f * scalingFactor;
    const ImVec2 windowPos = ImGui::GetWindowPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    ImVec2 blockPos = ImVec2(windowPos.x + (blockWidth * 0.5f), windowPos.y + timerWindowHeight - (blockHeight * 2.0f));
    drawList->AddRectFilled(blockPos,
                            ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight),
                            ImGui::ColorConvertFloat4ToU32(ColorMap.at("debug_red")));  // Red
    blockPos.x += blockWidth;
    drawList->AddRectFilled(blockPos,
                            ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight),
                            ImGui::ColorConvertFloat4ToU32(ColorMap.at("debug_green")));  // Green
    blockPos.x += blockWidth;
    drawList->AddRectFilled(blockPos,
                            ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight),
                            ImGui::ColorConvertFloat4ToU32(ColorMap.at("debug_blue")));  // Blue
    blockPos.x += blockWidth;
    drawList->AddRectFilled(blockPos,
                            ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight),
                            ImGui::ColorConvertFloat4ToU32(ColorMap.at("debug_yellow")));  // Yellow
    blockPos.x += blockWidth;
    drawList->AddRectFilled(blockPos,
                            ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight),
                            ImGui::ColorConvertFloat4ToU32(ColorMap.at("debug_white")));  // White
    blockPos.x += blockWidth;
    drawList->AddRectFilled(blockPos,
                            ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight),
                            ImGui::ColorConvertFloat4ToU32(ColorMap.at("debug_black")));  // Black

    ImGui::End();

    return Result::SUCCESS;
}

Result DefaultCompositor::renderDebugViewport() {
    if (!debugViewportEnabled) {
        return Result::SUCCESS;
    }

    const ImGuiIO& io = ImGui::GetIO();

    if (!ImGui::Begin("Viewport Debug", nullptr, ImGuiWindowFlags_NoResize)) {
        ImGui::End();
        return Result::SUCCESS;
    }

    const F32 mainWindowWidth = io.DisplaySize.x;
    const F32 mainWindowHeight = io.DisplaySize.y;
    ImGui::Text("Window Size: %.2f x %.2f", mainWindowWidth, mainWindowHeight);

    const F32 framebufferWidth = io.DisplayFramebufferScale.x;
    const F32 framebufferHeight = io.DisplayFramebufferScale.y;
    ImGui::Text("Framebuffer Scale: %.2f x %.2f", framebufferWidth, framebufferHeight);

    ImGui::Text("Render Window Scale: %.2f", scalingFactor);

    ImGui::End();

    return Result::SUCCESS;
}

Result DefaultCompositor::renderDebugDemo() {
    if (!debugDemoEnabled) {
        return Result::SUCCESS;
    }

    ImGui::ShowDemoWindow();

    return Result::SUCCESS;
}

Result DefaultCompositor::renderSeparator() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float separatorHeight = 2.0f * scalingFactor;
    const ImVec4 separatorColor = ColorMap.at("separator");

    const ImVec2 rectMin = ImVec2(viewport->Pos.x, viewport->Pos.y + currentHeight);
    const ImVec2 rectMax = ImVec2(viewport->Pos.x + viewport->Size.x, viewport->Pos.y + currentHeight + separatorHeight);

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    drawList->AddRectFilled(rectMin, rectMax, ImGui::ColorConvertFloat4ToU32(separatorColor));

    currentHeight += separatorHeight;

    return Result::SUCCESS;
}

Result DefaultCompositor::renderDockspace() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();

    const ImVec2 dockspacePos = ImVec2(viewport->Pos.x, viewport->Pos.y + currentHeight);
    const ImVec2 dockspaceSize = ImVec2(viewport->Size.x, viewport->Size.y - currentHeight);

    mainDockspaceID = ImHashStr("MainDockSpace");

    ImGui::SetNextWindowPos(dockspacePos);
    ImGui::SetNextWindowSize(dockspaceSize);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
                                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                   ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("DockSpace", nullptr, windowFlags);
    ImGui::PopStyleVar(3);

    if (particles.empty()) {
        const int particleCount = 60;
        particles.reserve(particleCount);

        std::srand(static_cast<unsigned int>(std::time(nullptr)));

        for (int i = 0; i < particleCount; ++i) {
            Particle particle;
            particle.position = ImVec2(
                dockspacePos.x + static_cast<F32>(std::rand()) / RAND_MAX * dockspaceSize.x,
                dockspacePos.y + static_cast<F32>(std::rand()) / RAND_MAX * dockspaceSize.y
            );
            particle.velocity = 10.0f + static_cast<F32>(std::rand()) / RAND_MAX * 30.0f;
            particle.radius = 1.0f + static_cast<F32>(std::rand()) / RAND_MAX * 2.5f;
            particle.alpha = 0.15f + static_cast<F32>(std::rand()) / RAND_MAX * 0.25f;
            particle.phase = static_cast<F32>(std::rand()) / RAND_MAX * 6.28318f; // Random phase 0-2π
            particles.push_back(particle);
        }
    }

    ImGui::DockSpace(mainDockspaceID, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);

    // TODO: Stop rendering particles when the view is hidden.

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    const F32 time = static_cast<F32>(ImGui::GetTime());
    const F32 deltaTime = ImGui::GetIO().DeltaTime;

    for (auto& particle : particles) {
        particle.position.x += particle.velocity * 0.15f * deltaTime * scalingFactor;
        particle.position.y -= particle.velocity * deltaTime * scalingFactor;

        F32 waveOffset = std::sin(time * 0.5f + particle.phase) * 2.0f * scalingFactor;
        ImVec2 renderPos = ImVec2(particle.position.x + waveOffset, particle.position.y);

        if (renderPos.y < dockspacePos.y - 10.0f) {
            particle.position.y = dockspacePos.y + dockspaceSize.y + 10.0f;
            particle.position.x = dockspacePos.x + static_cast<F32>(std::rand()) / RAND_MAX * dockspaceSize.x;
        }
        if (renderPos.x > dockspacePos.x + dockspaceSize.x + 10.0f) {
            particle.position.x = dockspacePos.x - 10.0f;
        }

        F32 twinkle = std::sin(time * 2.0f + particle.phase) * 0.1f + 0.9f;
        F32 finalAlpha = particle.alpha * twinkle;

        ImVec4 baseColor = ColorMap.at("cyber_blue");
        ImU32 color = ImGui::ColorConvertFloat4ToU32(ImVec4(baseColor.x, baseColor.y, baseColor.z, finalAlpha));
        ImU32 glowColor = ImGui::ColorConvertFloat4ToU32(ImVec4(baseColor.x, baseColor.y, baseColor.z, finalAlpha * 0.3f));

        drawList->AddCircleFilled(renderPos, particle.radius * scalingFactor * 2.5f, glowColor, 12);
        drawList->AddCircleFilled(renderPos, particle.radius * scalingFactor, color, 12);
    }

    ImGui::End();

    return Result::SUCCESS;
}

Result DefaultCompositor::renderDetachedSurfaces() {
    for (const auto& [flowgraphId, flowgraph] : flowgraphs) {
        if (!blocksCache.contains(flowgraphId)){
            continue;
        }
        const auto& blocks = blocksCache[flowgraphId];

        for (const auto& [blockName, blockPtr] : blocks) {
            if (!blockPtr ||
                !blockPtr->interface() ||
                blockPtr->state() == Block::State::Destroying ||
                blockPtr->state() == Block::State::Destroyed) {
                continue;
            }

            for (const auto& surface : blockPtr->surfaces()) {
                for (const auto& manifest : surface->manifests()) {
                    if (!manifest.surface || manifest.surface->raw() == 0) {
                        continue;
                    }

                    const std::string surfaceMetaKey = "surface_" + manifest.id;
                    SurfaceMeta surfaceMeta;
                    flowgraph->getMeta(surfaceMetaKey, surfaceMeta, blockName);

                    if (!surfaceMeta.detached) {
                        continue;
                    }

                    const std::string windowId = flowgraphId + ":" + blockName + ":" + manifest.id;
                    const std::string windowTitle = jst::fmt::format("{} ({})###{}", blockPtr->config().title(), blockName, windowId);

                    ImGui::SetNextWindowSize(ImVec2(static_cast<float>(surfaceMeta.detachedWidth) * scalingFactor,
                                                    static_cast<float>(surfaceMeta.detachedHeight) * scalingFactor), ImGuiCond_FirstUseEver);

                    const float windowPadding = 3.0f * scalingFactor;
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f * scalingFactor);
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(windowPadding, windowPadding));

                    bool windowOpen = true;
                    if (ImGui::Begin(windowTitle.c_str(), &windowOpen, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
                        const auto availableRegion = ImGui::GetContentRegionAvail();
                        if (helperCheckSurfaceResize(surface,
                                                     manifest,
                                                     availableRegion,
                                                     surfaceMeta.detachedWidth,
                                                     surfaceMeta.detachedHeight,
                                                     scalingFactor,
                                                     true)) {
                            flowgraph->setMeta(surfaceMetaKey, surfaceMeta, blockName);
                        }

                        const ImVec2 surfaceSize(availableRegion.x, availableRegion.y);
                        helperRenderSurfaceContent(surface, manifest, surfaceSize, 0.0f, true);
                    }
                    ImGui::End();

                    ImGui::PopStyleVar(2);

                    if (!windowOpen) {
                        surfaceMeta.detached = false;
                        flowgraph->setMeta(surfaceMetaKey, surfaceMeta, blockName);

                        helperSurfaceResize(surface,
                                            static_cast<U64>(surfaceMeta.attachedWidth * ImGui::GetIO().DisplayFramebufferScale.x),
                                            static_cast<U64>(surfaceMeta.attachedHeight * ImGui::GetIO().DisplayFramebufferScale.y),
                                            scalingFactor,
                                            false);
                    }
                }
            }
        }
    }

    return Result::SUCCESS;
}

Result DefaultCompositor::renderDocumentations() {
    std::vector<std::string> toRemove;

    for (auto& [docKey, open] : openDocumentations) {
        if (!open) {
            toRemove.push_back(docKey);
            continue;
        }

        const auto parts = Parser::SplitString(docKey, ":");
        if (parts.size() < 2) {
            toRemove.push_back(docKey);
            continue;
        }

        const std::string& flowgraphId = parts[0];
        const std::string& blockName = parts[1];

        if (!flowgraphs.contains(flowgraphId)) {
            toRemove.push_back(docKey);
            continue;
        }

        const auto& blocks = blocksCache[flowgraphId];
        if (!blocks.contains(blockName)) {
            toRemove.push_back(docKey);
            continue;
        }

        const auto& blockPtr = blocks.at(blockName);
        const std::string windowTitle = jst::fmt::format("{} Documentation ({})###{}", blockPtr->config().title(), blockName, docKey);

        ImGui::SetNextWindowSize(ImVec2(500 * scalingFactor, 400 * scalingFactor), ImGuiCond_FirstUseEver);

        if (ImGui::Begin(windowTitle.c_str(), &open)) {
            const auto& description = blockPtr->config().description();
            ImGui::Markdown(description.c_str(), description.length(), markdownConfig);
        }
        ImGui::End();
    }

    for (const auto& key : toRemove) {
        openDocumentations.erase(key);
    }

    return Result::SUCCESS;
}

Result DefaultCompositor::renderStacks() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();

    std::vector<std::string> stacksToRemove;
    for (auto& [stack, state] : stacks) {
        auto& [enabled, id] = state;

        if (!enabled) {
            stacksToRemove.push_back(stack);
            continue;
        }

        ImGuiWindowClass windowClass;
        windowClass.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoCloseButton | ImGuiDockNodeFlags_NoWindowMenuButton;
        ImGui::SetNextWindowClass(&windowClass);

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowDockID(mainDockspaceID, ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(ImVec2(500.0f * scalingFactor, 300.0f * scalingFactor), ImGuiCond_FirstUseEver);
        ImGui::Begin(stack.c_str(), &enabled);
        ImGui::PopStyleVar();

        bool isDockNew = false;

        if (!id) {
            isDockNew = true;
            id = ImGui::GetID(jst::fmt::format("##Stack{}", stack).c_str());
        }

        ImGui::DockSpace(id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);

        if (isDockNew && stack == "Graph") {
            ImGuiID dock_id_main;

            ImGui::DockBuilderRemoveNode(id);
            ImGui::DockBuilderAddNode(id);
            ImGui::DockBuilderSetNodePos(id, ImVec2(viewport->Pos.x, currentHeight));
            ImGui::DockBuilderSetNodeSize(id, ImVec2(viewport->Size.x, viewport->Size.y - currentHeight));

            dock_id_main = id;
            ImGui::DockBuilderDockWindow("Flowgraph", dock_id_main);

            ImGui::DockBuilderFinish(id);
        }

        ImGui::End();
    }

    for (const auto& stack : stacksToRemove) {
        stacks.erase(stack);
    }

    return Result::SUCCESS;
}

const std::unordered_map<std::string, std::unordered_map<std::string, ImVec4>> DefaultCompositor::themes = {
    {
        "Dark", {
            // Core Brand Colors
            {"cyber_blue", ImVec4(0.30f, 0.75f, 0.95f, 1.0f)},
            {"accent_color", ImVec4(0.90f, 0.65f, 0.50f, 1.00f)},
            {"accent_active", ImVec4(1.00f, 0.85f, 0.70f, 1.00f)},

            // Welcome Card Icon Colors
            {"welcome_icon_new", ImVec4(0.30f, 0.75f, 0.95f, 1.0f)},
            {"welcome_icon_open", ImVec4(0.30f, 0.75f, 0.95f, 1.0f)},
            {"welcome_icon_examples", ImVec4(0.30f, 0.75f, 0.95f, 1.0f)},

            // Status Colors
            {"success_green", ImVec4(0.0f, 1.0f, 0.0f, 1.0f)},
            {"warning_yellow", ImVec4(1.0f, 0.8f, 0.0f, 1.0f)},
            {"error_red", ImVec4(1.0f, 0.0f, 0.0f, 1.0f)},
            {"info_blue", ImVec4(0.0f, 0.0f, 1.0f, 1.0f)},

            // Background Colors
            {"background", ImVec4(0.0f, 0.0f, 0.0f, 1.00f)},
            {"panel", ImVec4(0.05f, 0.05f, 0.05f, 1.00f)},
            {"card", ImVec4(0.08f, 0.08f, 0.08f, 1.00f)},
            {"popup_bg", ImVec4(0.06f, 0.06f, 0.06f, 1.0f)},
            {"modal_dim", ImVec4(0.00f, 0.00f, 0.00f, 0.60f)},
            {"notification_bg", ImVec4(0.08f, 0.08f, 0.08f, 0.90f)},

            // Button Colors
            {"action_btn", ImVec4(0.20f, 0.47f, 0.96f, 1.0f)},
            {"action_btn_hovered", ImVec4(0.25f, 0.52f, 0.99f, 1.0f)},
            {"action_btn_active", ImVec4(0.15f, 0.35f, 0.85f, 1.0f)},

            // Text Colors
            {"text_primary", ImVec4(0.90f, 0.90f, 0.90f, 1.00f)},
            {"text_secondary", ImVec4(0.50f, 0.50f, 0.52f, 1.00f)},
            {"text_disabled", ImVec4(0.40f, 0.40f, 0.40f, 1.0f)},
            {"text_selected_bg", ImVec4(0.25f, 0.25f, 0.27f, 0.50f)},

            // Border Colors
            {"border", ImVec4(0.18f, 0.18f, 0.18f, 0.75f)},
            {"border_shadow", ImVec4(0.00f, 0.00f, 0.00f, 0.00f)},

            // Frame Colors
            {"frame_bg_hovered", ImVec4(0.10f, 0.10f, 0.10f, 0.95f)},
            {"frame_bg_active", ImVec4(0.12f, 0.12f, 0.12f, 0.95f)},
            {"modal_input_bg", ImVec4(0.09f, 0.09f, 0.09f, 1.0f)},

            // Title Bar Colors
            {"title_bg_active", ImVec4(0.06f, 0.06f, 0.06f, 0.97f)},
            {"title_bg_collapsed", ImVec4(0.04f, 0.04f, 0.04f, 0.80f)},

            // Scrollbar Colors
            {"scrollbar_bg", ImVec4(0.0f, 0.0f, 0.0f, 0.0f)},
            {"scrollbar_grab", ImVec4(0.20f, 0.20f, 0.20f, 0.80f)},
            {"scrollbar_grab_hovered", ImVec4(0.28f, 0.28f, 0.28f, 0.80f)},
            {"scrollbar_grab_active", ImVec4(0.35f, 0.35f, 0.35f, 0.80f)},

            // Button Colors
            {"button", ImVec4(0.10f, 0.10f, 0.10f, 1.00f)},
            {"button_hovered", ImVec4(0.12f, 0.12f, 0.12f, 1.00f)},
            {"button_active", ImVec4(0.14f, 0.14f, 0.14f, 1.00f)},

            // Header Colors
            {"header", ImVec4(0.10f, 0.10f, 0.10f, 0.80f)},
            {"header_hovered", ImVec4(0.15f, 0.15f, 0.15f, 0.80f)},
            {"header_active", ImVec4(0.20f, 0.20f, 0.20f, 0.80f)},

            // Separator Colors
            {"separator", ImVec4(0.15f, 0.15f, 0.15f, 0.50f)},
            {"separator_hovered", ImVec4(0.25f, 0.25f, 0.25f, 0.80f)},
            {"separator_active", ImVec4(0.35f, 0.35f, 0.35f, 1.00f)},

            // Resize Grip Colors
            {"resize_grip", ImVec4(0.18f, 0.18f, 0.18f, 0.60f)},
            {"resize_grip_hovered", ImVec4(0.28f, 0.28f, 0.28f, 0.80f)},
            {"resize_grip_active", ImVec4(0.38f, 0.38f, 0.38f, 1.00f)},

            // Tab Colors
            {"tab", ImVec4(0.04f, 0.04f, 0.04f, 0.95f)},
            {"tab_hovered", ImVec4(0.12f, 0.12f, 0.12f, 1.00f)},
            {"tab_selected", ImVec4(0.08f, 0.08f, 0.08f, 0.95f)},
            {"tab_dimmed", ImVec4(0.03f, 0.03f, 0.03f, 0.95f)},
            {"tab_dimmed_selected", ImVec4(0.06f, 0.06f, 0.06f, 0.95f)},

            // Docking Colors
            {"docking_preview", ImVec4(0.30f, 0.30f, 0.30f, 0.50f)},
            {"docking_empty_bg", ImVec4(0.02f, 0.02f, 0.02f, 1.00f)},

            // Plot Colors
            {"plot_lines", ImVec4(0.50f, 0.50f, 0.52f, 1.00f)},
            {"plot_lines_hovered", ImVec4(0.60f, 0.60f, 0.62f, 1.00f)},
            {"plot_histogram", ImVec4(0.40f, 0.40f, 0.42f, 1.00f)},
            {"plot_histogram_hovered", ImVec4(0.50f, 0.50f, 0.52f, 1.00f)},

            // Table Colors
            {"table_header_bg", ImVec4(0.06f, 0.06f, 0.06f, 0.95f)},
            {"table_border_strong", ImVec4(0.15f, 0.15f, 0.15f, 0.70f)},
            {"table_border_light", ImVec4(0.10f, 0.10f, 0.10f, 0.50f)},
            {"table_row_bg", ImVec4(0.00f, 0.00f, 0.00f, 0.00f)},
            {"table_row_bg_alt", ImVec4(1.00f, 1.00f, 1.00f, 0.02f)},

            // Drag & Drop Colors
            {"drag_drop_target", ImVec4(0.50f, 0.50f, 0.52f, 0.90f)},

            // Navigation Colors
            {"nav_windowing_highlight", ImVec4(1.00f, 1.00f, 1.00f, 0.50f)},
            {"nav_windowing_dim_bg", ImVec4(0.00f, 0.00f, 0.00f, 0.60f)},

            // Node Editor Colors
            {"cell_background", ImVec4(0.03f, 0.03f, 0.03f, 0.90f)},
            {"node_background", ImVec4(0.05f, 0.05f, 0.05f, 1.0f)},
            {"node_outline", ImVec4(0.878f, 0.573f, 0.0f, 1.0f)},
            {"node_outline_error", ImVec4(0.86f, 0.24f, 0.24f, 1.0f)},
            {"node_outline_pending", ImVec4(0.50f, 0.50f, 0.50f, 1.0f)},
            {"node_title_bar", ImVec4(0.0f, 0.0f, 0.0f, 0.0f)},
            {"node_pin", ImVec4(0.878f, 0.573f, 0.0f, 1.0f)},
            {"node_link", ImVec4(0.878f, 0.573f, 0.0f, 1.0f)},
            {"grid_line", ImVec4(0.12f, 0.12f, 0.13f, 1.0f)},
            {"grid_background", ImVec4(0.0f, 0.0f, 0.0f, 0.0f)},

            // Debug Colors
            {"debug_red", ImVec4(1.0f, 0.0f, 0.0f, 1.0f)},
            {"debug_green", ImVec4(0.0f, 1.0f, 0.0f, 1.0f)},
            {"debug_blue", ImVec4(0.0f, 0.0f, 1.0f, 1.0f)},
            {"debug_yellow", ImVec4(1.0f, 1.0f, 0.0f, 1.0f)},
            {"debug_white", ImVec4(1.0f, 1.0f, 1.0f, 1.0f)},
            {"debug_black", ImVec4(0.0f, 0.0f, 0.0f, 1.0f)},

            // Selection Colors
            {"selection_lime", ImVec4(0.5f, 1.0f, 0.0f, 1.0f)},
        }
    }
};

std::shared_ptr<Compositor::Impl> DefaultCompositorFactory() {
    return std::make_shared<DefaultCompositor>();
}

}  // namespace Jetstream
