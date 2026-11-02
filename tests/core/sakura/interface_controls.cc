#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "compositor/default/presenters/flowgraph/config.hh"
#include "compositor/default/views/flowgraph/editor/config/base.hh"
#include "typist.hh"

#include <any>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

using namespace Jetstream;

TEST_CASE("Shared config presenter preserves descriptors values and Python diagnostics", "[core][sakura][interface]") {
    Flowgraph::View::BlockData block;
    block.config = {{"count", U64{9007199254740993ULL}}, {"code", "pass"}};
    block.interfaceConfigs = {
        {.name = "count", .format = {{"type", "uint"}}},
        {.name = "code", .format = {{"type", "python"}}},
        {.name = "console", .format = {{"type", "python-console"}}},
    };
    Runtime::Context::Diagnostic diagnostic;
    diagnostic.status = "ready";
    diagnostic.healthy = true;
    diagnostic.console = {"output"};
    block.metrics.push_back({.name = "python", .format = {{"type", "python-diagnostic"}, {"visibility", "internal"}},
                             .value = diagnostic});
    const auto node = BuildFlowgraphConfigFields("node", block);
    const auto detached = BuildFlowgraphConfigFields("window", block);
    REQUIRE(node.size() == 3);
    REQUIRE(detached.size() == 3);
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
    REQUIRE(node[2].status == "ready");
    REQUIRE(node[2].statusTone == Sakura::StatusTone::Success);
    REQUIRE(node[2].consoleOutput == diagnostic.console);
}

TEST_CASE("Controls report decoding failures rather than silently displaying defaults", "[core][sakura][interface]") {
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
    REQUIRE(field.isSimple());
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

TEST_CASE("Vector controls recover their cached values after a decoding error", "[core][sakura][interface]") {
    FlowgraphConfigVectorField field;
    FlowgraphConfigFieldConfig config{
        .id = "vector",
        .name = "items",
        .format = {{"type", "vector"}, {"value_type", "float"}},
        .values = {{"items", std::vector<F32>{1.0f, 2.0f}}},
    };
    REQUIRE(field.update(config) == Result::SUCCESS);
    auto invalid = config;
    invalid.values["items"] = std::string("invalid");
    REQUIRE(field.update(invalid) == Result::ERROR);
    REQUIRE(field.update(config) == Result::SUCCESS);
    config.format["value_type"] = std::string("uint");
    config.values.clear();
    REQUIRE(field.update(config) == Result::SUCCESS);
}

TEST_CASE("Retained unsigned range edits reject unsafe conversions and recover on the next edit",
          "[core][sakura][interface][range-input]") {
    const auto [text, valid, expected] = GENERATE(
        std::tuple{"-1", false, U64{0}},
        std::tuple{"-0.1", false, U64{0}},
        std::tuple{"nan", false, U64{0}},
        std::tuple{"inf", false, U64{0}},
        std::tuple{"-inf", false, U64{0}},
        std::tuple{"1e999", false, U64{0}},
        std::tuple{"1e30", false, U64{0}},
        std::tuple{"18446744073709551615", false, U64{0}},
        std::tuple{"18446744073709551616", false, U64{0}},
        std::tuple{"0", true, U64{0}},
        std::tuple{"3.5", true, U64{4}},
        std::tuple{"300", true, U64{300}},
        std::tuple{"18446742974197923840", true, U64{18446742974197923840ULL}});
    const U64 initial = GENERATE(U64{8}, std::numeric_limits<U64>::max());
    CAPTURE(text, initial);

    SakuraTest::Typist typist;
    FlowgraphConfigRangeField field;
    std::vector<U64> applied;
    std::vector<std::string> errors;
    REQUIRE(field.update({
        .id = "averaging",
        .name = "averaging",
        .label = "Averaging",
        .format = {{"type", "range"}, {"min", 1.0f}, {"max", 256.0f}, {"value_type", "uint"}},
        .values = {{"averaging", initial}},
        .onApply = [&](Parser::Map patch, bool, std::function<void(Result)>) { applied.push_back(std::any_cast<U64>(patch.at("averaging"))); },
        .onError = [&](Result result, std::string message) {
            CHECK(result == Result::ERROR);
            errors.push_back(std::move(message));
        },
    }) == Result::SUCCESS);
    SakuraTest::Host host(field.component());
    typist.root = &host;
    typist.place = [&] { host.place(typist.ctx, {0.0f, 0.0f, 320.0f, 100.0f}); };
    typist.frame();
    const auto inputs = SakuraTest::Find<Sakura::NodeRangeInput>(field.component());
    REQUIRE(inputs.size() == 1);
    const auto bounds = SakuraTest::Peek::Frame(*inputs.front());
    const auto edit = [&](const char* value) {
        REQUIRE(host.send(SakuraTest::Mouse(MouseEventType::Click, bounds.x + 30.0f,
                                          bounds.y + bounds.height * 0.5f, MouseButton::Left, 2)));
        typist.frame();
        REQUIRE(SakuraTest::FocusOwner(host) != nullptr);
        typist.type(value);
        typist.press(KeyCode::Enter);
    };
    edit(text);
    if (valid) {
        REQUIRE(applied == std::vector<U64>{expected});
        CHECK(errors.empty());
    } else {
        if ((std::string(text) == "nan" || std::string(text) == "inf") && initial == 8) {
            CHECK(applied == std::vector<U64>{initial});
        } else {
            CHECK(applied.empty());
        }
        const std::string typed = text;
        const bool silent = typed == "-inf" || typed == "1e999" ||
                            ((typed == "nan" || typed == "inf") && initial == 8);
        REQUIRE(errors.size() == (silent ? 0u : 1u));
        for (const auto& error : errors) {
            CHECK(error.find("Averaging") != std::string::npos);
        }
        applied.clear();
        errors.clear();
        edit("16");
        REQUIRE(applied == std::vector<U64>{16});
        CHECK(errors.empty());
    }
}

TEST_CASE("Retained float range edits keep finite values outside the slider bounds",
          "[core][sakura][interface][range-input]") {
    const char* text = GENERATE("-0.25", "2.5", "nan", "inf", "-inf", "1e999");
    CAPTURE(text);
    SakuraTest::Typist typist;
    FlowgraphConfigRangeField field;
    std::vector<F32> applied;
    REQUIRE(field.update({
        .id = "range",
        .name = "range",
        .format = {{"type", "range"}, {"min", 0.0f}, {"max", 1.0f}},
        .values = {{"range", 0.5f}},
        .onApply = [&](Parser::Map patch, bool, std::function<void(Result)>) { applied.push_back(std::any_cast<F32>(patch.at("range"))); },
    }) == Result::SUCCESS);
    SakuraTest::Host host(field.component());
    typist.root = &host;
    typist.place = [&] { host.place(typist.ctx, {0.0f, 0.0f, 320.0f, 100.0f}); };
    typist.frame();
    const auto inputs = SakuraTest::Find<Sakura::NodeRangeInput>(field.component());
    REQUIRE(inputs.size() == 1);
    const auto bounds = SakuraTest::Peek::Frame(*inputs.front());
    REQUIRE(host.send(SakuraTest::Mouse(MouseEventType::Click, bounds.x + 30.0f,
                                      bounds.y + bounds.height * 0.5f, MouseButton::Left, 2)));
    typist.frame();
    REQUIRE(SakuraTest::FocusOwner(host) != nullptr);
    typist.type(text);
    typist.press(KeyCode::Enter);
    if (std::string(text) == "-0.25" || std::string(text) == "2.5") {
        REQUIRE(applied == std::vector<F32>{std::stof(text)});
    } else if (std::string(text) == "nan" || std::string(text) == "inf") {
        CHECK(applied == std::vector<F32>{0.5f});
    } else {
        CHECK(applied.empty());
    }
}

TEST_CASE("Range fields keep the newest submitted value until its configuration edit completes",
          "[core][sakura][interface][range-input]") {
    const auto outcome = GENERATE(Result::SUCCESS, Result::ERROR, Result::WARNING);
    CAPTURE(outcome);
    SakuraTest::Typist typist;
    FlowgraphConfigRangeField field;
    std::vector<F32> applied;
    std::vector<std::function<void(Result)>> completions;
    FlowgraphConfigFieldConfig config{
        .id = "gain",
        .name = "gain",
        .label = "Gain",
        .format = {{"type", "range"}, {"min", 0.0f}, {"max", 100.0f}},
        .values = {{"gain", 100.0f}},
        .onApply = [&](Parser::Map patch, bool, std::function<void(Result)> complete) {
            applied.push_back(std::any_cast<F32>(patch.at("gain")));
            completions.push_back(std::move(complete));
        },
    };
    SakuraTest::Host host(field.component());
    typist.root = &host;
    typist.place = [&] {
        REQUIRE(field.update(config) == Result::SUCCESS);
        host.place(typist.ctx, {0.0f, 0.0f, 320.0f, 100.0f});
    };
    typist.frame();
    const auto inputs = SakuraTest::Find<Sakura::NodeRangeInput>(field.component());
    REQUIRE(inputs.size() == 1);
    const auto bounds = SakuraTest::Peek::Frame(*inputs.front());
    const auto shown = [&]() -> std::string {
        for (auto* label : SakuraTest::Find<Sakura::Label>(*inputs.front())) {
            for (const auto& instance : label->config().instances) {
                if (instance.visible && !instance.str.empty()) {
                    return instance.str;
                }
            }
        }
        return {};
    };
    const auto click = [&](F32 fraction) {
        const F32 x = bounds.x + bounds.width * fraction;
        const F32 y = bounds.y + bounds.height * 0.5f;
        REQUIRE(host.send(SakuraTest::Mouse(MouseEventType::Click, x, y)));
        typist.frame();
        REQUIRE(host.send(SakuraTest::Mouse(MouseEventType::Release, x, y)));
        typist.frame();
    };

    REQUIRE(shown() == "100");
    click(0.5f);
    REQUIRE(applied.size() == 1);
    REQUIRE(applied[0] == Catch::Approx(50.0f));
    for (int frame = 0; frame < 10; ++frame) {
        typist.frame();
        REQUIRE(shown() == "50");
    }

    click(0.3f);
    REQUIRE(applied.size() == 2);
    REQUIRE(applied[1] == Catch::Approx(30.0f));
    config.values["gain"] = 48.0f;
    completions[0](Result::SUCCESS);
    for (int frame = 0; frame < 10; ++frame) {
        typist.frame();
        REQUIRE(shown() == "30");
    }

    const F32 resolved = outcome == Result::SUCCESS ? 28.0f : 48.0f;
    config.values["gain"] = resolved;
    completions[1](outcome);
    for (int frame = 0; frame < 10; ++frame) {
        typist.frame();
        REQUIRE(shown() == (outcome == Result::SUCCESS ? "28" : "48"));
    }
    REQUIRE(applied.size() == 2);
}

TEST_CASE("Dropdowns return strings and distinguish duplicate display labels",
          "[core][sakura][interface]") {
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
        .onApply = [&](Parser::Map patch, bool, std::function<void(Result)>) { edits.push_back(std::move(patch)); },
    };
    REQUIRE(Block::Interface::ValidateFormat(config.format) == Result::SUCCESS);
    REQUIRE(field.update(config) == Result::SUCCESS);
    const auto combos = SakuraTest::Find<Sakura::NodeCombo>(field.component());
    REQUIRE(combos.size() == 1);
    REQUIRE(combos.front()->config().selectedIndex == std::optional<U64>{0});
    combos.front()->config().onSelect(1);
    REQUIRE(edits.size() == 1);
    REQUIRE(std::any_cast<std::string>(edits.front().at("deviceString")) == expected);

    config.values["deviceString"] = expected;
    REQUIRE(field.update(config) == Result::SUCCESS);
    CHECK(combos.front()->config().selectedIndex == std::optional<U64>{1});

    options.pop_back();
    config.format["options"] = options;
    REQUIRE(field.update(config) == Result::SUCCESS);
    CHECK_FALSE(combos.front()->config().selectedIndex.has_value());
    CHECK(edits.size() == 1);
}

TEST_CASE("String dropdown options work with boolean config values",
          "[core][sakura][interface]") {
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
        .onApply = [&](Parser::Map patch, bool, std::function<void(Result)>) { edits.push_back(std::move(patch)); },
    }) == Result::SUCCESS);
    const auto combos = SakuraTest::Find<Sakura::NodeCombo>(field.component());
    REQUIRE(combos.size() == 1);
    CHECK(combos.front()->config().selectedIndex == std::optional<U64>{0});
    combos.front()->config().onSelect(1);
    REQUIRE(edits.size() == 1);
    REQUIRE(std::any_cast<std::string>(edits.front().at("forward")) == "false");
    bool forward = true;
    REQUIRE(Parser::Deserialize(edits.front(), "forward", forward) == Result::SUCCESS);
    REQUIRE_FALSE(forward);
}

TEST_CASE("Numeric fields apply explicit scale independently of their unit label",
          "[core][sakura][interface]") {
    SakuraTest::Typist typist;
    FlowgraphConfigFloatField field;
    std::vector<Parser::Map> edits;
    REQUIRE(field.update({
        .id = "scaled",
        .name = "value",
        .format = {{"type", "float"}, {"unit", "MHz"}, {"scale", 10.0f}, {"precision", 3}},
        .values = {{"value", 20.0f}},
        .onApply = [&](Parser::Map patch, bool, std::function<void(Result)>) { edits.push_back(std::move(patch)); },
    }) == Result::SUCCESS);
    SakuraTest::Host host(field.component());
    typist.root = &host;
    typist.place = [&] { host.place(typist.ctx, {0.0f, 0.0f, 320.0f, 100.0f}); };
    typist.frame();
    const auto inputs = SakuraTest::Find<Sakura::NodeFloatInput>(field.component());
    REQUIRE(inputs.size() == 1);
    const auto bounds = SakuraTest::Peek::Frame(*inputs.front());
    REQUIRE(host.click(bounds.x + 30.0f, bounds.y + 10.0f));
    typist.frame();
    REQUIRE(SakuraTest::FocusOwner(host) != nullptr);
    typist.type("3.25");
    typist.press(KeyCode::Enter);
    REQUIRE(edits.size() == 1);
    CHECK(std::any_cast<F32>(edits.front().at("value")) == 32.5f);
}
