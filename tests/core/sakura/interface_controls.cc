#include <catch2/catch_test_macros.hpp>

#include "compositor/default/presenters/flowgraph/config.hh"
#include "compositor/default/views/flowgraph/editor/config/dropdown.hh"
#include "compositor/default/views/flowgraph/editor/config/float.hh"
#include "compositor/default/views/flowgraph/editor/config/base.hh"
#include "compositor/default/views/flowgraph/editor/metrics/table.hh"
#include "harness.hh"
#include <imgui_internal.h>

using namespace Jetstream;

namespace {

void SelectOption(SakuraTest::HeadlessUi& ui, FlowgraphConfigDropdownField& field, U64 index) {
    ImRect combo;
    const auto frame = [&] {
        ui.frame([&] {
            field.render(ui.sakura());
            combo = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
        });
    };
    frame();
    ui.setMouse(combo.GetCenter(), false);
    frame();
    ui.setMouse(combo.GetCenter(), true);
    frame();
    ui.setMouse(combo.GetCenter(), false);
    frame();
    frame();
    const auto& popups = ImGui::GetCurrentContext()->OpenPopupStack;
    REQUIRE_FALSE(popups.empty());
    const auto* popup = popups.back().Window;
    REQUIRE(popup != nullptr);
    const ImVec2 position(popup->DC.CursorStartPos.x + 20.0f,
                          popup->DC.CursorStartPos.y + (index + 0.5f) * ImGui::GetFontSize());
    ui.setMouse(position, false);
    frame();
    ui.setMouse(position, true);
    frame();
    ui.setMouse(position, false);
    frame();
}

}  // namespace

TEST_CASE("Dropdowns return strings and distinguish duplicate display labels",
          "[core][sakura][interface]") {
    SakuraTest::HeadlessUi ui(1.0f, {800.0f, 600.0f});
    FlowgraphConfigDropdownField field;
    const std::string expected = "driver=lime,serial=abc,remote=tcp://localhost:55132";
    Parser::Sequence options{
        Parser::Map{{"label", "Radio, (USB) α"}, {"value", ""}},
        Parser::Map{{"label", "Radio, (USB) α"}, {"value", expected}},
    };
    std::vector<Parser::Map> edits;
    FlowgraphConfigFieldConfig config{
        .id = "device",
        .name = "deviceString",
        .format = {{"type", "dropdown"}, {"options", options}},
        .values = {{"deviceString", ""}},
        .onApply = [&](Parser::Map patch, bool) { edits.push_back(std::move(patch)); },
    };
    REQUIRE(Block::Interface::ValidateFormat(config.format) == Result::SUCCESS);
    REQUIRE(field.update(config) == Result::SUCCESS);
    SelectOption(ui, field, 1);
    REQUIRE(edits.size() == 1);
    REQUIRE(std::any_cast<std::string>(edits.front().at("deviceString")) == expected);

    config.values["deviceString"] = expected;
    options.pop_back();
    config.format["options"] = options;
    REQUIRE(field.update(config) == Result::SUCCESS);
    ui.frame([&] { field.render(ui.sakura()); });
    REQUIRE(edits.size() == 1);
}

TEST_CASE("Shared config presenter preserves descriptors values and Python diagnostics",
          "[core][sakura][interface]") {
    Flowgraph::View::BlockData block;
    block.config = {{"count", U64{9007199254740993ULL}}, {"code", "pass"}};
    block.interfaceConfigs = {
        {.name = "count", .format = {{"type", "uint"}}},
        {.name = "code", .format = {{"type", "python"}}},
    };
    Runtime::Context::Diagnostic diagnostic;
    diagnostic.status = "ready";
    diagnostic.healthy = true;
    diagnostic.console = {"output"};
    block.metrics.push_back({.name = "python", .format = {{"type", "python-diagnostic"}, {"visibility", "internal"}},
                             .value = diagnostic});
    const auto node = BuildFlowgraphConfigFields("node", block);
    const auto detached = BuildFlowgraphConfigFields("window", block);
    REQUIRE(node.size() == 2);
    REQUIRE(detached.size() == 2);
    U64 count = 0;
    REQUIRE(Parser::Deserialize(node[0].values, node[0].name, count) == Result::SUCCESS);
    REQUIRE(count == U64{9007199254740993ULL});
    REQUIRE(node[0].values == detached[0].values);
    REQUIRE(node[0].format == detached[0].format);
    std::string code;
    REQUIRE(Parser::Deserialize(node[1].values, node[1].name, code) == Result::SUCCESS);
    REQUIRE(code == "pass");
    REQUIRE(node[1].status == "ready");
    REQUIRE(detached[1].consoleOutput == diagnostic.console);
    REQUIRE(detached[1].consoleVisible);
}

TEST_CASE("String dropdown options work with boolean config values",
          "[core][sakura][interface]") {
    SakuraTest::HeadlessUi ui;
    FlowgraphConfigDropdownField field;
    std::vector<Parser::Map> edits;
    REQUIRE(field.update({
        .id = "direction",
        .name = "forward",
        .format = {{"type", "dropdown"}, {"options", Parser::Sequence{
            Parser::Map{{"label", "Forward"}, {"value", "true"}},
            Parser::Map{{"label", "Inverse"}, {"value", "false"}},
        }}},
        .values = {{"forward", true}},
        .onApply = [&](Parser::Map patch, bool) { edits.push_back(std::move(patch)); },
    }) == Result::SUCCESS);
    SelectOption(ui, field, 1);
    REQUIRE(edits.size() == 1);
    REQUIRE(std::any_cast<std::string>(edits.front().at("forward")) == "false");
    bool forward = true;
    REQUIRE(Parser::Deserialize(edits.front(), "forward", forward) == Result::SUCCESS);
    REQUIRE_FALSE(forward);
}

TEST_CASE("Controls report decoding failures rather than silently displaying defaults",
          "[core][sakura][interface]") {
    SakuraTest::HeadlessUi ui;
    FlowgraphConfigFieldInstance field;
    std::vector<std::string> errors;
    FlowgraphConfigFieldConfig config{
        .id = "numeric",
        .name = "amount",
        .format = {{"type", "float"}},
        .values = {{"amount", "invalid"}},
        .onError = [&](Result result, std::string message) {
            REQUIRE(result == Result::ERROR);
            errors.push_back(std::move(message));
        },
    };
    field.update(config);
    REQUIRE(errors.size() == 1);
    REQUIRE(errors.front().find("amount") != std::string::npos);
    for (int i = 0; i < 5; ++i) {
        field.update(config);
    }
    REQUIRE(errors.size() == 1);
    config.values["amount"] = std::string("still invalid");
    field.update(config);
    REQUIRE(errors.size() == 1);
    ui.frame([&] {
        ImGui::LogToBuffer();
        field.render(ui.sakura());
        const std::string text = ImGui::GetCurrentContext()->LogBuffer.c_str();
        ImGui::LogFinish();
        REQUIRE(text.find("Invalid configuration value") != std::string::npos);
    });
    config.values["amount"] = 12.0f;
    field.update(config);
    REQUIRE(field.isSimple());
    REQUIRE(errors.size() == 1);
    config.values["amount"] = std::string("invalid again");
    field.update(config);
    REQUIRE(errors.size() == 2);
    field.update(config);
    REQUIRE(errors.size() == 2);
    config.id = "another-field";
    field.update(config);
    REQUIRE(errors.size() == 3);
}

TEST_CASE("Numeric fields apply explicit scale independently of their unit label",
          "[core][sakura][interface]") {
    SakuraTest::HeadlessUi ui;
    FlowgraphConfigFloatField field;
    std::vector<Parser::Map> edits;
    field.update({
        .id = "scaled",
        .name = "value",
        .format = {{"type", "float"}, {"unit", "MHz"}, {"scale", 10.0f}, {"precision", 3}},
        .values = {{"value", 20.0f}},
        .onApply = [&](Parser::Map patch, bool) { edits.push_back(std::move(patch)); },
    });
    ImRect rect;
    const auto frame = [&] {
        ui.frame([&] {
            field.render(ui.sakura());
            rect = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
        });
    };
    frame();
    ui.setMouse(rect.GetCenter(), false);
    frame();
    ui.setMouse(rect.GetCenter(), true);
    frame();
    ui.setMouse(rect.GetCenter(), false);
    frame();
    auto* input = ImGui::GetInputTextState(ImGui::GetActiveID());
    REQUIRE(input != nullptr);
    REQUIRE(std::string(input->TextA.Data) == "2.000");
    input->SelectAll();
    ImGui::GetIO().AddInputCharactersUTF8("3.25");
    frame();
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Enter, true);
    frame();
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Enter, false);
    frame();
    REQUIRE(edits.size() == 1);
    REQUIRE(std::any_cast<F32>(edits.front().at("value")) == 32.5f);
}

TEST_CASE("Vector controls recover their cached values after a decoding error",
          "[core][sakura][interface]") {
    SakuraTest::HeadlessUi ui;
    FlowgraphConfigVectorField field;
    FlowgraphConfigFieldConfig config{
        .id = "vector",
        .name = "items",
        .format = {{"type", "vector"}, {"value_type", "float"}},
        .values = {{"items", std::vector<F32>{1.0f, 2.0f}}},
    };
    const auto height = [&] {
        F32 result = 0.0f;
        ui.frame([&] {
            const auto start = ImGui::GetCursorPosY();
            field.render(ui.sakura());
            result = ImGui::GetCursorPosY() - start;
        });
        return result;
    };
    REQUIRE(field.update(config) == Result::SUCCESS);
    const F32 originalHeight = height();
    REQUIRE(originalHeight > 0.0f);
    auto invalid = config;
    invalid.values["items"] = std::string("invalid");
    REQUIRE(field.update(invalid) == Result::ERROR);
    REQUIRE(field.update(config) == Result::SUCCESS);
    REQUIRE(height() == originalHeight);
    config.format["value_type"] = std::string("uint");
    config.values.clear();
    REQUIRE(field.update(config) == Result::SUCCESS);
    REQUIRE(height() == 0.0f);
}

TEST_CASE("Table metrics accept structured cells containing tabs and newlines",
          "[core][sakura][interface]") {
    SakuraTest::HeadlessUi ui;
    FlowgraphMetricTable metric;
    Parser::Map value;
    REQUIRE(Parser::Serialize(value, "columns", std::vector<std::string>{"Name", "Value"}) == Result::SUCCESS);
    REQUIRE(Parser::Serialize(value, "rows", std::vector<std::vector<std::string>>{
        {"Line\nbreak", "Tab\tinside"}, {"Comma, colon:", "(parentheses)"},
    }) == Result::SUCCESS);
    for (int frame = 0; frame < 3; ++frame) {
        metric.update({.id = "table", .format = {{"type", "table"}}, .value = value});
        ui.frame([&] {
            ImGui::LogToBuffer();
            metric.render(ui.sakura());
            const std::string text = ImGui::GetCurrentContext()->LogBuffer.c_str();
            ImGui::LogFinish();
            REQUIRE(text.find("Comma, colon:") != std::string::npos);
            REQUIRE(text.find("Tab\tinside") != std::string::npos);
            REQUIRE(text.find("Invalid") == std::string::npos);
        });
    }
}
