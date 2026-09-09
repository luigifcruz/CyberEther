#include <catch2/catch_test_macros.hpp>
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
    }

    void frame(const FlowgraphConfigRangeField& field) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("range-input-test", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        field.render(ctx);
        inputRect = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
        ImGui::End();
        ImGui::Render();
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
    Sakura::Context ctx;
    ImRect inputRect;
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
        .onApply = [&](Parser::Map patch, bool) {
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
        .onApply = [&](Parser::Map patch, bool) {
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
