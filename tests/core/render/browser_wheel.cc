#include <catch2/catch_test_macros.hpp>

#include "../../../src/viewport/glfw/tools/browser_wheel.hh"

using namespace Jetstream;

namespace {

using Classifier = Viewport::BrowserWheel::Classifier;

Classifier::Sample Pixels(F64 time, F64 deltaY, F64 deltaX = 0.0) {
    return {.time = time, .pixels = true, .deltaX = deltaX, .deltaY = deltaY};
}

}  // namespace

TEST_CASE("Browser wheels treat fractional pixel deltas as precise and line deltas as notched",
          "[core][render][browser_wheel]") {
    Classifier classifier;
    CHECK(classifier.classify(Pixels(0.0, 3.5)));
    CHECK(classifier.classify(Pixels(16.0, 0.0, 2.0)));
    CHECK_FALSE(classifier.classify({.time = 1000.0, .pixels = false, .deltaY = 3.0}));
}

TEST_CASE("Browser wheels never treat a pinch as a precise scroll", "[core][render][browser_wheel]") {
    Classifier classifier;
    CHECK_FALSE(classifier.classify({.time = 0.0, .pixels = true, .pinch = true, .deltaY = 1.5}));
}

TEST_CASE("Browser wheels read whole notch-sized pixel deltas as a mouse wheel unless a gesture is running",
          "[core][render][browser_wheel]") {
    Classifier classifier;
    CHECK_FALSE(classifier.classify(Pixels(0.0, 100.0)));
    CHECK_FALSE(classifier.classify(Pixels(10.0, -53.0)));
    CHECK(classifier.classify(Pixels(20.0, 49.0)));

    CHECK(classifier.classify(Pixels(500.0, 2.5)));
    CHECK(classifier.classify(Pixels(560.0, 100.0)));
    CHECK(classifier.classify(Pixels(650.0, 100.0)));
    CHECK_FALSE(classifier.classify(Pixels(760.0, 100.0)));
}

TEST_CASE("Browser wheels scale precise deltas into port units", "[core][render][browser_wheel]") {
    const Extent2D<F32> scaled = Classifier::Normalize({1.5f, -2.0f});
    CHECK(scaled.x == 15.0f);
    CHECK(scaled.y == -20.0f);
}
