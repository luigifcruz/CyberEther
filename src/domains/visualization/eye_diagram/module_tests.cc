#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <limits>
#include <string>

#include "jetstream/domains/visualization/eye_diagram/module.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

TEST_CASE("Eye Diagram module carries interpolation and folding across buffers",
          "[modules][eye_diagram]") {
    TestContext ctx("eye_diagram", DeviceType::CPU, RuntimeType::NATIVE, "generic");

    Modules::EyeDiagram config;
    config.samplesPerSymbol = 2;
    config.interpolation = 2;
    config.rangeMin = 0.0f;
    config.rangeMax = 8.0f;
    ctx.setConfig(config);

    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {3}) == Result::SUCCESS);
    ctx.setInput("signal", input);
    REQUIRE(ctx.start() == Result::SUCCESS);

    for (U64 buffer = 0; buffer < 3; ++buffer) {
        for (U64 i = 0; i < 3; ++i) {
            input.data<F32>()[i] = static_cast<F32>(buffer * 3 + i);
        }
        REQUIRE(ctx.compute() == Result::SUCCESS);

        const auto& output = ctx.output("signal");
        REQUIRE(output.size() == 9);
        for (U64 i = 0; i < output.size(); ++i) {
            const F32 expected = buffer == 0 ? 0.5f :
                (static_cast<F32>(i) * 0.5f + (buffer == 2 ? 4.0f : 0.0f)) / 8.0f;
            CHECK(output.data<F32>()[i] == Catch::Approx(expected));
        }
    }

    REQUIRE(ctx.stop() == Result::SUCCESS);
}

TEST_CASE("Eye Diagram module selects IQ components and clips amplitudes",
          "[modules][eye_diagram]") {
    for (const std::string component : {"real", "imag"}) {
        DYNAMIC_SECTION("Component: " << component) {
            TestContext ctx("eye_diagram", DeviceType::CPU, RuntimeType::NATIVE, "generic");

            Modules::EyeDiagram config;
            config.samplesPerSymbol = 1;
            config.interpolation = 1;
            config.component = component;
            config.rangeMin = -1.0f;
            config.rangeMax = 1.0f;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {3}) == Result::SUCCESS);
            input.data<CF32>()[0] = {-2.0f, 2.0f};
            input.data<CF32>()[1] = {0.0f, 0.0f};
            input.data<CF32>()[2] = {2.0f, -2.0f};
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const F32* output = ctx.output("signal").data<F32>();
            CHECK(output[0] == (component == "real" ? 0.0f : 1.0f));
            CHECK(output[1] == 0.5f);
            CHECK(output[2] == (component == "real" ? 1.0f : 0.0f));
        }
    }
}

TEST_CASE("Eye Diagram module resets incomplete windows after non-finite samples",
          "[modules][eye_diagram]") {
    TestContext ctx("eye_diagram", DeviceType::CPU, RuntimeType::NATIVE, "generic");

    Modules::EyeDiagram config;
    config.samplesPerSymbol = 1;
    config.interpolation = 2;
    config.rangeMin = 0.0f;
    config.rangeMax = 4.0f;
    ctx.setConfig(config);

    const F32 samples[] = {
        0.0f, 1.0f, std::numeric_limits<F32>::quiet_NaN(), 2.0f, 3.0f, 4.0f,
    };

    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {6}) == Result::SUCCESS);
    for (U64 i = 0; i < 6; ++i) {
        input.data<F32>()[i] = samples[i];
    }
    ctx.setInput("signal", input);
    REQUIRE(ctx.run() == Result::SUCCESS);

    const F32* output = ctx.output("signal").data<F32>();
    for (U64 i = 0; i < 5; ++i) {
        CHECK(output[i] == Catch::Approx(0.5f + static_cast<F32>(i) * 0.125f));
    }
}

TEST_CASE("Eye Diagram module rejects invalid configuration",
          "[modules][eye_diagram][validation]") {
    for (U64 invalid = 0; invalid < 6; ++invalid) {
        DYNAMIC_SECTION("Case: " << invalid) {
            TestContext ctx("eye_diagram", DeviceType::CPU, RuntimeType::NATIVE, "generic");

            Modules::EyeDiagram config;
            if (invalid == 0) {
                config.samplesPerSymbol = 0;
            } else if (invalid == 1) {
                config.samplesPerSymbol = 257;
            } else if (invalid == 2) {
                config.interpolation = 3;
            } else if (invalid == 3) {
                config.rangeMax = config.rangeMin;
            } else if (invalid == 4) {
                config.component = "magnitude";
            } else if (invalid == 5) {
                config.rangeMin = std::numeric_limits<F32>::infinity();
            }
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {32}) == Result::SUCCESS);
            ctx.setInput("signal", input);
            CHECK(ctx.run() == Result::ERROR);
        }
    }
}
