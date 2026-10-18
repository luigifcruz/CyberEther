#include <catch2/catch_test_macros.hpp>

#include "harness.hh"
#include "render/sakura/detail/text_metrics.hh"

#include <memory>
#include <string>
#include <vector>

using namespace Jetstream;

namespace {

class MetricsWindow : public SakuraTest::FakeWindow {
 public:
    void install(const std::string& name) {
        fonts[name] = std::make_shared<Render::Components::Font>(Render::Components::Font::Config{});
    }

    void erase(const std::string& name) {
        fonts.erase(name);
    }
};

}  // namespace

TEST_CASE("Text widths reuse cached results and separate font, size, and text",
          "[core][sakura][text-metrics]") {
    MetricsWindow window;
    window.install("default_mono");
    window.install("other");
    Sakura::TextMetrics metrics;
    metrics.setWindow(&window);

    CHECK(metrics.measure("default_mono", "hello", 16.0f) == 0.0f);
    REQUIRE(metrics.widths.size() == 1);
    metrics.widths.begin()->second = 123.0f;
    CHECK(metrics.measure("default_mono", "hello", 16.0f) == 123.0f);

    CHECK(metrics.measure("other", "hello", 16.0f) == 0.0f);
    CHECK(metrics.measure("default_mono", "hello", 17.0f) == 0.0f);
    CHECK(metrics.measure("default_mono", "world", 16.0f) == 0.0f);
    CHECK(metrics.measure("default_mono", std::string("hello\0", 6), 16.0f) == 0.0f);
    CHECK(metrics.measure("default_mono", "", 16.0f) == 0.0f);
    CHECK(metrics.measure("", "hello", 16.0f) == 0.0f);
    CHECK(metrics.measure("missing", "hello", 16.0f) == 0.0f);
    CHECK(metrics.widths.size() == 8);
    CHECK(metrics.measure("default_mono", "hello", 16.0f) == 123.0f);
    CHECK(metrics.widths.size() == 8);
}

TEST_CASE("Text width keys preserve binary font and text boundaries",
          "[core][sakura][text-metrics]") {
    MetricsWindow window;
    window.install("default_mono");
    Sakura::TextMetrics metrics;
    metrics.setWindow(&window);
    const F32 size = 16.0f;
    const auto* font = metrics.label("")->getConfig().font.get();
    const std::string fields = std::string(1, '\0') +
        std::string(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& boundary : {fields, fields + std::string(reinterpret_cast<const char*>(&font), sizeof(font))}) {
        metrics.widths.clear();
        CHECK(metrics.measure("mono", boundary + "payload", size) == 0.0f);
        REQUIRE(metrics.widths.size() == 1);
        metrics.widths.begin()->second = 42.0f;
        CHECK(metrics.measure("mono" + boundary, "payload", size) == 0.0f);
        CHECK(metrics.widths.size() == 2);
        CHECK(metrics.measure("mono", boundary + "payload", size) == 42.0f);
    }
}

TEST_CASE("Text width capacity clears on a miss but not a hit",
          "[core][sakura][text-metrics]") {
    MetricsWindow window;
    window.install("default_mono");
    Sakura::TextMetrics metrics;
    metrics.setWindow(&window);
    for (U64 i = 0; i < 256; ++i) {
        CHECK(metrics.measure("", std::to_string(i), 16.0f) == 0.0f);
        CHECK(metrics.widths.size() == i + 1);
    }
    for (auto& [key, width] : metrics.widths) {
        width = 31.0f;
    }
    CHECK(metrics.measure("", "0", 16.0f) == 31.0f);
    CHECK(metrics.widths.size() == 256);
    CHECK(metrics.measure("", "256", 16.0f) == 0.0f);
    REQUIRE(metrics.widths.size() == 1);
    metrics.widths.begin()->second = 32.0f;
    CHECK(metrics.measure("", "256", 16.0f) == 32.0f);
    CHECK(metrics.measure("", "0", 16.0f) == 0.0f);
    CHECK(metrics.widths.size() == 2);
}

TEST_CASE("Text metric windows retain or clear both caches together",
          "[core][sakura][text-metrics]") {
    MetricsWindow first;
    MetricsWindow second;
    first.install("default_mono");
    second.install("default_mono");
    Sakura::TextMetrics metrics;
    metrics.setWindow(&first);
    CHECK(metrics.measure("", "hello", 16.0f) == 0.0f);
    REQUIRE(metrics.widths.size() == 1);
    REQUIRE(metrics.labels.size() == 1);
    metrics.widths.begin()->second = 17.0f;
    const auto* label = metrics.label("");

    metrics.setWindow(&first);
    CHECK(metrics.label("") == label);
    CHECK(metrics.measure("", "hello", 16.0f) == 17.0f);
    CHECK(metrics.widths.size() == 1);
    metrics.setWindow(&second);
    CHECK(metrics.labels.empty());
    CHECK(metrics.widths.empty());
    CHECK(metrics.measure("", "hello", 16.0f) == 0.0f);
    REQUIRE(metrics.widths.size() == 1);
    metrics.setWindow(nullptr);
    CHECK(metrics.labels.empty());
    CHECK(metrics.widths.empty());
    CHECK(metrics.measure("", "hello", 16.0f) == 0.0f);
    CHECK(metrics.advances("", "hello", 16.0f) == std::vector<F32>(5, 0.0f));
    CHECK(metrics.labels.empty());
    CHECK(metrics.widths.empty());
}

TEST_CASE("Text widths track fallback availability without caching failed measurements",
          "[core][sakura][text-metrics]") {
    MetricsWindow window;
    Sakura::TextMetrics metrics;
    CHECK(metrics.measure("named", "hello", 16.0f) == 0.0f);
    metrics.setWindow(&window);
    CHECK(metrics.measure("named", "hello", 16.0f) == 0.0f);
    CHECK(metrics.advances("named", "hello", 16.0f) == std::vector<F32>(5, 0.0f));
    CHECK(metrics.labels.empty());
    CHECK(metrics.widths.empty());

    window.install("default_mono");
    CHECK(metrics.measure("named", "hello", 16.0f) == 0.0f);
    REQUIRE(metrics.widths.size() == 1);
    metrics.widths.begin()->second = 19.0f;
    CHECK(metrics.label("named") == metrics.label("default_mono"));
    CHECK(metrics.measure("named", "hello", 16.0f) == 19.0f);

    window.install("named");
    CHECK(metrics.measure("named", "hello", 16.0f) == 0.0f);
    CHECK(metrics.widths.size() == 2);
    CHECK(metrics.label("named") != metrics.label("default_mono"));
    window.erase("named");
    CHECK(metrics.measure("named", "hello", 16.0f) == 19.0f);
    window.erase("default_mono");
    CHECK(metrics.measure("named", "hello", 16.0f) == 0.0f);
    CHECK(metrics.widths.size() == 2);
}

TEST_CASE("Per-byte text advances do not consult or populate the width cache",
          "[core][sakura][text-metrics]") {
    MetricsWindow window;
    window.install("default_mono");
    Sakura::TextMetrics metrics;
    metrics.setWindow(&window);
    const std::string value("a\xc3\xa9\n\t\0z", 7);
    CHECK(metrics.advances("", value, 16.0f) == std::vector<F32>(value.size(), 0.0f));
    CHECK(metrics.widths.empty());
    CHECK(metrics.measure("", value, 16.0f) == 0.0f);
    REQUIRE(metrics.widths.size() == 1);
    metrics.widths.begin()->second = 77.0f;
    CHECK(metrics.advances("", value, 16.0f) == std::vector<F32>(value.size(), 0.0f));
    CHECK(metrics.widths.size() == 1);
    CHECK(metrics.measure("", value, 16.0f) == 77.0f);
    CHECK(metrics.advances("", "", 16.0f).empty());
}
