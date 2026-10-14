#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <imgui.h>
#include <imgui_internal.h>

#include <any>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "compositor/default/views/flowgraph/editor/config/range.hh"
#include "render/sakura/context.hh"

using namespace Jetstream;

namespace {

class RangeInputWindow final : public Render::Window {
 public:
    RangeInputWindow() : Window(Config{}) { _scalingFactor = 1.0f; }

    const Stats& stats() const override { return windowStats; }
    std::string info() const override { return "RangeInputWindow"; }
    constexpr DeviceType device() const override { return DeviceType::None; }

 protected:
    Result bindSurface(const std::shared_ptr<Render::Surface>&) override {
        return Result::SUCCESS;
    }
    Result unbindSurface(const std::shared_ptr<Render::Surface>&) override {
        return Result::SUCCESS;
    }
    Result underlyingCreate() override { return Result::SUCCESS; }
    Result underlyingDestroy() override { return Result::SUCCESS; }
    Result underlyingBegin() override { return Result::SUCCESS; }
    Result underlyingEnd() override { return Result::SUCCESS; }
    Result underlyingSynchronize() override { return Result::SUCCESS; }

 private:
    Stats windowStats{};
};

class RangeInputUi {
 public:
    RangeInputUi() : ui(ImGui::CreateContext(), ImGui::DestroyContext) {
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
        io.DisplaySize = ImVec2(500.0f, 300.0f);
        io.DeltaTime = 1.0f / 60.0f;
        io.Fonts->AddFontDefault();
        ctx.render = &window;
        ctx.palette = palette;
    }

    template<typename Field>
    void frame(const Field& field) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("range-input-test", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        field.render(ctx);
        inputRect = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
        F32 minimum = FLT_MAX;
        F32 maximum = -FLT_MAX;
        for (const auto& vertex : ImGui::GetWindowDrawList()->VtxBuffer) {
            if (vertex.col == IM_COL32(255, 0, 255, 255)) {
                minimum = std::min(minimum, vertex.pos.x);
                maximum = std::max(maximum, vertex.pos.x);
            }
        }
        knobFraction = ((minimum + maximum) * 0.5f - inputRect.Min.x) / inputRect.GetWidth();
        ImGui::End();
        ImGui::Render();
    }

    F32 move(const F32 fraction) {
        const F32 x = std::floor(inputRect.Min.x + fraction * inputRect.GetWidth());
        ImGui::GetIO().AddMousePosEvent(x, inputRect.GetCenter().y);
        return (x - inputRect.Min.x) / inputRect.GetWidth();
    }

    F32 position() const {
        return knobFraction;
    }

    void beginEdit(const FlowgraphConfigRangeField& field) {
        ImGuiIO& io = ImGui::GetIO();
        io.DeltaTime = io.MouseDoubleClickTime + 0.01f;
        frame(field);
        io.DeltaTime = 1.0f / 60.0f;
        const ImVec2 center = inputRect.GetCenter();
        io.AddMousePosEvent(center.x, center.y);
        frame(field);
        for (const bool down : {true, false, true, false}) {
            io.AddMouseButtonEvent(ImGuiMouseButton_Left, down);
            frame(field);
        }
        frame(field);
        frame(field);
        REQUIRE(ImGui::GetIO().WantTextInput);
    }

    void submit(const FlowgraphConfigRangeField& field, const char* text) {
        ImGuiIO& io = ImGui::GetIO();
        io.AddInputCharactersUTF8(text);
        frame(field);
        io.AddKeyEvent(ImGuiKey_Enter, true);
        frame(field);
        io.AddKeyEvent(ImGuiKey_Enter, false);
        frame(field);
    }

 private:
    std::unique_ptr<ImGuiContext, decltype(&ImGui::DestroyContext)> ui;
    RangeInputWindow window;
    Sakura::Palette palette{{"frame_bg_active", {1.0f, 0.0f, 1.0f, 1.0f}}};
    Sakura::Context ctx;
    ImRect inputRect;
    F32 knobFraction = 0.0f;
};

struct UnsignedRangeCase {
    const char* text;
    bool valid;
    U64 expected = 0;
};

}  // namespace

TEST_CASE("Unsigned range edits reject unsafe conversions before applying config",
          "[core][render][sakura][range-input][validation]") {
    const auto test = GENERATE(
        UnsignedRangeCase{"-1", false},
        UnsignedRangeCase{"-0.1", false},
        UnsignedRangeCase{"nan", false},
        UnsignedRangeCase{"inf", false},
        UnsignedRangeCase{"-inf", false},
        UnsignedRangeCase{"1e999", false},
        UnsignedRangeCase{"1e30", false},
        UnsignedRangeCase{"18446744073709551615", false},
        UnsignedRangeCase{"18446744073709551616", false},
        UnsignedRangeCase{"0", true, 0},
        UnsignedRangeCase{"3.5", true, 4},
        UnsignedRangeCase{"300", true, 300},
        UnsignedRangeCase{"18446742974197923840", true, 18446742974197923840ULL});
    const char* initial = GENERATE("8", "18446744073709551615");
    CAPTURE(test.text, initial);

    RangeInputUi ui;
    FlowgraphConfigRangeField field;
    std::vector<U64> applied;
    std::vector<std::string> errors;
    field.update({
        .id = "averaging",
        .name = "averaging",
        .label = "Averaging",
        .format = {{"type", "range"}, {"min", 1.0f}, {"max", 256.0f}, {"unit", "samples"}, {"value_type", "uint"}},
        .values = {{"averaging", static_cast<U64>(std::stoull(initial))}},
        .onApply = [&](Parser::Map patch, bool, std::function<void(Result)>) {
            applied.push_back(std::any_cast<U64>(patch.at("averaging")));
        },
        .onError = [&](Result result, std::string message) {
            CHECK(result == Result::ERROR);
            errors.push_back(std::move(message));
        },
    });
    ui.beginEdit(field);
    applied.clear();
    errors.clear();
    ui.submit(field, test.text);

    if (test.valid) {
        REQUIRE(applied.size() == 1);
        CHECK(applied.front() == test.expected);
        CHECK(errors.empty());
    } else {
        CHECK(applied.empty());
        if (std::strtof(test.text, nullptr) == std::strtof(initial, nullptr)) {
            CHECK(errors.empty());
        } else {
            REQUIRE(errors.size() == 1);
            CHECK(errors.front().find("Averaging") != std::string::npos);
        }

        ui.beginEdit(field);
        applied.clear();
        errors.clear();
        ui.submit(field, "16");
        REQUIRE(applied.size() == 1);
        CHECK(applied.front() == 16);
        CHECK(errors.empty());
    }
}

TEST_CASE("Float range edits preserve finite values outside the slider range",
          "[core][render][sakura][range-input][validation]") {
    const char* text = GENERATE("-0.25", "2.5", "nan", "inf", "-inf", "1e999");
    CAPTURE(text);
    const bool valid = std::string(text) == "-0.25" || std::string(text) == "2.5";

    RangeInputUi ui;
    FlowgraphConfigRangeField field;
    std::vector<F32> applied;
    std::vector<std::string> errors;
    field.update({
        .id = "range",
        .name = "range",
        .label = "Range",
        .format = {{"type", "range"}, {"min", 0.0f}, {"max", 1.0f}},
        .values = {{"range", 0.5f}},
        .onApply = [&](Parser::Map patch, bool, std::function<void(Result)>) {
            applied.push_back(std::any_cast<F32>(patch.at("range")));
        },
        .onError = [&](Result result, std::string message) {
            CHECK(result == Result::ERROR);
            errors.push_back(std::move(message));
        },
    });
    ui.beginEdit(field);
    applied.clear();
    errors.clear();
    ui.submit(field, text);

    if (valid) {
        REQUIRE(applied.size() == 1);
        CHECK(applied.front() == std::stof(text));
        CHECK(errors.empty());
    } else {
        CHECK(applied.empty());
        REQUIRE(errors.size() == 1);
    }
}

TEST_CASE("Range drags ignore stale and snapped backend values while held still",
          "[core][render][sakura][range-input][drag]") {
    RangeInputUi ui;
    Sakura::NodeRangeInput input;
    std::vector<F32> applied;
    Sakura::NodeRangeInput::Config config{
        .id = "gain",
        .min = 0.0f,
        .max = 100.0f,
        .value = 25.0f,
        .onChange = [&](const F32 value) { applied.push_back(value); },
    };
    input.update(config);
    ui.frame(input);
    const F32 target = ui.move(0.617f);
    ui.frame(input);
    ImGui::GetIO().AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    ui.frame(input);
    REQUIRE(ui.position() == Catch::Approx(target));
    REQUIRE(applied.size() == 1);
    REQUIRE(applied.front() == Catch::Approx(target * 100.0f));
    for (int frame = 0; frame < 8; ++frame) {
        input.update(config);
        ui.frame(input);
    }
    REQUIRE(applied.size() == 1);
    REQUIRE(applied.front() == Catch::Approx(target * 100.0f));

    for (const F32 echo : {25.0f, 60.0f, 61.0f}) {
        config.value = echo;
        for (int frame = 0; frame < 20; ++frame) {
            input.update(config);
            ui.frame(input);
            REQUIRE(ui.position() == Catch::Approx(target));
        }
    }
    ImGui::GetIO().DisplaySize.x = 600.0f;
    ui.frame(input);
    REQUIRE(ui.position() == Catch::Approx(target));
    REQUIRE(applied.size() == 1);

    ImGui::GetIO().AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    ui.frame(input);
    REQUIRE(applied.size() == 1);
    ui.frame(input);
    REQUIRE(ui.position() == Catch::Approx(0.61f));
}

TEST_CASE("Range drags submit changes immediately and preserve the final pointer position on release",
          "[core][render][sakura][range-input][drag]") {
    RangeInputUi ui;
    Sakura::NodeRangeInput input;
    std::vector<F32> applied;
    input.update({
        .id = "gain",
        .min = 0.0f,
        .max = 100.0f,
        .value = 25.0f,
        .onChange = [&](const F32 value) { applied.push_back(value); },
    });
    ui.frame(input);
    ui.move(0.3f);
    ui.frame(input);
    ImGui::GetIO().AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    ui.frame(input);
    REQUIRE(applied.size() == 1);
    for (const F32 fraction : {0.4f, 0.5f, 0.6f, 0.7f, 0.8f}) {
        const auto count = applied.size();
        const F32 target = ui.move(fraction);
        ui.frame(input);
        REQUIRE(applied.size() == count + 1);
        REQUIRE(applied.back() == Catch::Approx(target * 100.0f));
        for (int frame = 0; frame < 3; ++frame) {
            ui.frame(input);
        }
        REQUIRE(ui.position() == Catch::Approx(target));
        REQUIRE(applied.size() == count + 1);
    }
    const F32 target = ui.move(0.9f);
    ImGui::GetIO().AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    ui.frame(input);
    REQUIRE(applied.size() == 7);
    REQUIRE(applied.back() == Catch::Approx(target * 100.0f));
    for (int frame = 0; frame < 10; ++frame) {
        ui.frame(input);
    }
    REQUIRE(applied.size() == 7);
}

TEST_CASE("Integer range drags compare rounded values before submitting",
          "[core][render][sakura][range-input][drag]") {
    RangeInputUi ui;
    Sakura::NodeRangeInput input;
    std::vector<F32> applied;
    input.update({
        .id = "averaging",
        .min = 0.0f,
        .max = 10.0f,
        .value = 5.0f,
        .integer = true,
        .onChange = [&](const F32 value) { applied.push_back(value); },
    });
    ui.frame(input);
    ui.move(0.52f);
    ui.frame(input);
    ImGui::GetIO().AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    for (int frame = 0; frame < 20; ++frame) {
        ui.frame(input);
    }
    REQUIRE(applied.empty());
    REQUIRE(ui.position() == Catch::Approx(0.5f));
    ui.move(0.62f);
    for (int frame = 0; frame < 20; ++frame) {
        ui.frame(input);
    }
    REQUIRE(applied == std::vector<F32>{6.0f});
    REQUIRE(ui.position() == Catch::Approx(0.6f));
    ImGui::GetIO().AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    ui.frame(input);
    REQUIRE(applied == std::vector<F32>{6.0f});
}

TEST_CASE("Range clicks retain the newest value until its configuration edit completes",
          "[core][render][sakura][range-input][pending]") {
    const auto result = GENERATE(Result::SUCCESS, Result::ERROR, Result::WARNING);
    RangeInputUi ui;
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
    const auto frame = [&] {
        REQUIRE(field.update(config) == Result::SUCCESS);
        ui.frame(field);
    };
    const auto click = [&](const F32 fraction) {
        const F32 target = ui.move(fraction);
        frame();
        ImGui::GetIO().AddMouseButtonEvent(ImGuiMouseButton_Left, true);
        frame();
        ImGui::GetIO().AddMouseButtonEvent(ImGuiMouseButton_Left, false);
        frame();
        return target;
    };

    frame();
    frame();
    const F32 middle = click(0.5f);
    REQUIRE(applied.size() == 1);
    for (int i = 0; i < 60; ++i) {
        frame();
        REQUIRE(ui.position() == Catch::Approx(middle));
    }
    REQUIRE(applied.size() == 1);

    const F32 latest = click(0.3f);
    REQUIRE(applied.size() == 2);
    config.values["gain"] = 48.0f;
    completions[0](Result::SUCCESS);
    for (int i = 0; i < 60; ++i) {
        frame();
        REQUIRE(ui.position() == Catch::Approx(latest));
    }
    REQUIRE(applied.size() == 2);

    const F32 resolved = result == Result::SUCCESS ? 28.0f : 48.0f;
    config.values["gain"] = resolved;
    completions[1](result);
    for (int i = 0; i < 10; ++i) {
        frame();
        REQUIRE(ui.position() == Catch::Approx(resolved / 100.0f));
    }
    REQUIRE(applied.size() == 2);
}
