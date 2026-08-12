#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>

#include "jetstream/domains/dsp/agc/module.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

namespace {

void RequireAgcValidationError(const Registry::ModuleRegistration& impl,
                               const Modules::Agc& config,
                               const DataType dtype = DataType::F32) {
    Tensor input;
    REQUIRE(input.create(impl.device, dtype, {16}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("agc", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("AGC - Uses complex RMS power", "[modules][agc][cf32]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            Modules::Agc config;
            config.epsilon = 1e-30;
            ctx.setConfig(config);

            Tensor input(DeviceType::CPU, DataType::CF32, {4});
            for (U64 i = 0; i < input.size(); ++i) {
                input.at<CF32>(i) = CF32(3.0f, 4.0f);
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.dtype() == DataType::CF32);
            REQUIRE(out.shape(0) == 4);
            for (U64 i = 0; i < out.size(); ++i) {
                REQUIRE_THAT(out.at<CF32>(i).real(),
                             Catch::Matchers::WithinAbs(0.6f, 1e-6f));
                REQUIRE_THAT(out.at<CF32>(i).imag(),
                             Catch::Matchers::WithinAbs(0.8f, 1e-6f));
            }
        }
    }
}

TEST_CASE("AGC - Interpolates tiled gains and handles a partial tile",
          "[modules][agc][f32][tiles]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            Modules::Agc config;
            config.tileSize = 2;
            config.epsilon = 1e-30;
            ctx.setConfig(config);

            Tensor input(DeviceType::CPU, DataType::F32, {5});
            input.at<F32>(0) = 1.0f;
            input.at<F32>(1) = 1.0f;
            input.at<F32>(2) = 2.0f;
            input.at<F32>(3) = 2.0f;
            input.at<F32>(4) = 4.0f;

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE(out.shape(0) == 5);
            const F32 expected[] = {1.0f, 0.75f, 1.0f, 0.75f, 1.0f};
            for (U64 i = 0; i < out.size(); ++i) {
                REQUIRE_THAT(out.at<F32>(i),
                             Catch::Matchers::WithinAbs(expected[i], 1e-6f));
            }
        }
    }
}

TEST_CASE("AGC - Applies gain bounds", "[modules][agc][f32][gain]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            Modules::Agc config;
            config.tileSize = 1;
            config.epsilon = 1e-30;
            config.minGain = 0.1;
            config.maxGain = 10.0;
            config.maxGainChange = 1000.0;
            ctx.setConfig(config);

            Tensor input(DeviceType::CPU, DataType::F32, {2});
            input.at<F32>(0) = 0.001f;
            input.at<F32>(1) = 100.0f;

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(0.01f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1),
                         Catch::Matchers::WithinAbs(10.0f, 1e-6f));
        }
    }
}

TEST_CASE("AGC - Limits gain change between tiles",
          "[modules][agc][f32][gain][tiles]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            Modules::Agc config;
            config.tileSize = 2;
            config.epsilon = 1e-30;
            config.maxGain = 200.0;
            config.maxGainChange = 2.0;
            ctx.setConfig(config);

            Tensor input(DeviceType::CPU, DataType::F32, {4});
            input.at<F32>(0) = 1.0f;
            input.at<F32>(1) = 1.0f;
            input.at<F32>(2) = 0.01f;
            input.at<F32>(3) = 0.01f;

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1),
                         Catch::Matchers::WithinAbs(1.5f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2),
                         Catch::Matchers::WithinAbs(0.02f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(3),
                         Catch::Matchers::WithinAbs(0.02f, 1e-6f));
        }
    }
}

TEST_CASE("AGC - Processes sample lanes independently",
          "[modules][agc][f32][metadata]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            Modules::Agc config;
            config.tileSize = 2;
            config.epsilon = 1e-30;
            ctx.setConfig(config);

            Tensor input(DeviceType::CPU, DataType::F32, {4, 2});
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{0},
                .channel = Index{1},
            }) == Result::SUCCESS);
            input.at<F32>(0, 0) = 1.0f;
            input.at<F32>(1, 0) = 1.0f;
            input.at<F32>(2, 0) = 2.0f;
            input.at<F32>(3, 0) = 2.0f;
            input.at<F32>(0, 1) = 2.0f;
            input.at<F32>(1, 1) = 2.0f;
            input.at<F32>(2, 1) = 4.0f;
            input.at<F32>(3, 1) = 4.0f;

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            const F32 expected[] = {1.0f, 0.75f, 1.0f, 1.0f};
            for (U64 sample = 0; sample < 4; ++sample) {
                for (U64 lane = 0; lane < 2; ++lane) {
                    REQUIRE_THAT(out.at<F32>(sample, lane),
                                 Catch::Matchers::WithinAbs(
                                     expected[sample], 1e-6f));
                }
            }
        }
    }
}

TEST_CASE("AGC - Processes many multi-tile lanes",
          "[modules][agc][f32][metadata][tiles]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            Modules::Agc config;
            config.tileSize = 2;
            config.epsilon = 1e-30;
            config.minGain = 1e-9;
            ctx.setConfig(config);

            constexpr U64 channelCount = 1024;
            Tensor input(DeviceType::CPU, DataType::F32, {4, channelCount});
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{0},
                .channel = Index{1},
            }) == Result::SUCCESS);
            for (U64 channel = 0; channel < channelCount; ++channel) {
                const F32 amplitude = static_cast<F32>(1 + channel % 16);
                input.at<F32>(0, channel) = amplitude;
                input.at<F32>(1, channel) = amplitude;
                input.at<F32>(2, channel) = 2.0f * amplitude;
                input.at<F32>(3, channel) = 2.0f * amplitude;
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            const F32 expected[] = {1.0f, 0.75f, 1.0f, 1.0f};
            for (U64 sample = 0; sample < 4; ++sample) {
                for (U64 channel = 0; channel < channelCount; ++channel) {
                    REQUIRE_THAT(out.at<F32>(sample, channel),
                                 Catch::Matchers::WithinAbs(
                                     expected[sample], 1e-6f));
                }
            }
        }
    }
}

TEST_CASE("AGC - Keeps extreme finite CF32 input finite",
          "[modules][agc][cf32][limits]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            Modules::Agc config;
            config.epsilon = std::numeric_limits<F64>::denorm_min();
            config.minGain = std::numeric_limits<F64>::denorm_min();
            ctx.setConfig(config);

            const F32 max = std::numeric_limits<F32>::max();
            Tensor input(DeviceType::CPU, DataType::CF32, {1});
            input.at<CF32>(0) = CF32(0.75f * max, 0.5f * max);

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const CF32 sample = ctx.output("signal").at<CF32>(0);
            REQUIRE(std::isfinite(sample.real()));
            REQUIRE(std::isfinite(sample.imag()));
            REQUIRE_THAT(std::abs(sample),
                         Catch::Matchers::WithinAbs(1.0f, 1e-5f));
        }
    }
}

TEST_CASE("AGC - Saturates finite input instead of overflowing",
          "[modules][agc][f32][limits][tiles]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            Modules::Agc config;
            config.tileSize = 1;
            ctx.setConfig(config);

            const F32 max = std::numeric_limits<F32>::max();
            const F32 belowMax = std::nextafter(max, 0.0f);
            Tensor input(DeviceType::CPU, DataType::F32, {5});
            input.at<F32>(0) = 0.0f;
            input.at<F32>(1) = max;
            input.at<F32>(2) = -max;
            input.at<F32>(3) = belowMax;
            input.at<F32>(4) = -belowMax;

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.at<F32>(0) == 0.0f);
            for (U64 i = 1; i < out.size(); ++i) {
                REQUIRE(std::isfinite(out.at<F32>(i)));
                REQUIRE(std::abs(out.at<F32>(i)) <= max);
            }
            REQUIRE(out.at<F32>(1) > 0.0f);
            REQUIRE(out.at<F32>(2) < 0.0f);
            REQUIRE(out.at<F32>(3) > 0.0f);
            REQUIRE(out.at<F32>(4) < 0.0f);
        }
    }
}

TEST_CASE("AGC - Clamps F64 rounding above the F32 boundary",
          "[modules][agc][f32][limits][rounding]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            Modules::Agc config;
            config.tileSize = 1;
            config.epsilon = std::numeric_limits<F64>::denorm_min();
            config.maxGain = std::numeric_limits<F64>::max();
            ctx.setConfig(config);

            constexpr F32 boundarySample = 2.460624500599806e-06f;
            Tensor input(DeviceType::CPU, DataType::F32, {3});
            input.at<F32>(0) = 0.0f;
            input.at<F32>(1) = boundarySample;
            input.at<F32>(2) = -boundarySample;

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const F32 max = std::numeric_limits<F32>::max();
            const auto& out = ctx.output("signal");
            REQUIRE(out.at<F32>(1) == max);
            REQUIRE(out.at<F32>(2) == -max);
        }
    }
}

TEST_CASE("AGC - Complex saturation preserves phase",
          "[modules][agc][cf32][limits][tiles]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            Modules::Agc config;
            config.tileSize = 1;
            ctx.setConfig(config);

            const F32 max = std::numeric_limits<F32>::max();
            Tensor input(DeviceType::CPU, DataType::CF32, {2});
            input.at<CF32>(0) = CF32(0.0f, 0.0f);
            input.at<CF32>(1) = CF32(0.75f * max, -0.5f * max);

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const CF32 sample = ctx.output("signal").at<CF32>(1);
            REQUIRE(std::isfinite(sample.real()));
            REQUIRE(std::isfinite(sample.imag()));
            REQUIRE(std::isfinite(std::abs(sample)));
            REQUIRE(sample.real() > 0.0f);
            REQUIRE(sample.imag() < 0.0f);
            REQUIRE_THAT(sample.imag() / sample.real(),
                         Catch::Matchers::WithinAbs(-2.0f / 3.0f, 1e-6f));
        }
    }
}

TEST_CASE("AGC - Preserves representable CF32 magnitude",
          "[modules][agc][cf32][limits]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            Modules::Agc config;
            config.minGain = 1.0;
            config.maxGain = 1.0;
            ctx.setConfig(config);

            const F32 component = 0.6f * std::numeric_limits<F32>::max();
            Tensor input(DeviceType::CPU, DataType::CF32, {1});
            input.at<CF32>(0) = CF32(component, component);

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const CF32 sample = ctx.output("signal").at<CF32>(0);
            REQUIRE_THAT(sample.real(),
                         Catch::Matchers::WithinRel(component, 1e-6f));
            REQUIRE_THAT(sample.imag(),
                         Catch::Matchers::WithinRel(component, 1e-6f));
            REQUIRE(std::isfinite(std::abs(sample)));
        }
    }
}

TEST_CASE("AGC - Preserves silence", "[modules][agc][f32][zero]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            ctx.setConfig(Modules::Agc{});

            Tensor input(DeviceType::CPU, DataType::F32, {8});
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            for (U64 i = 0; i < out.size(); ++i) {
                REQUIRE(out.at<F32>(i) == 0.0f);
            }
        }
    }
}

TEST_CASE("AGC - Reconfigures controls without recreating output",
          "[modules][agc][f32][reconfigure]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("agc", impl.device, impl.runtime, impl.provider);
            Modules::Agc config;
            config.epsilon = 1e-30;
            ctx.setConfig(config);

            Tensor input(DeviceType::CPU, DataType::F32, {2});
            input.at<F32>(0) = 2.0f;
            input.at<F32>(1) = 2.0f;
            ctx.setInput("signal", input);

            REQUIRE(ctx.start() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            const auto outputId = ctx.output("signal").id();
            REQUIRE_THAT(ctx.output("signal").at<F32>(0),
                         Catch::Matchers::WithinAbs(1.0f, 1e-6f));

            config.tileSize = 1;
            config.reference = 0.5;
            REQUIRE(ctx.reconfigure(config) == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            REQUIRE(ctx.output("signal").id() == outputId);
            REQUIRE_THAT(ctx.output("signal").at<F32>(0),
                         Catch::Matchers::WithinAbs(0.5f, 1e-6f));
            REQUIRE(ctx.stop() == Result::SUCCESS);
        }
    }
}

TEST_CASE("AGC - Rejects invalid controls", "[modules][agc][validation]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::Agc config;

            SECTION("zero tile size") {
                config.tileSize = 0;
            }
            SECTION("non-positive reference") {
                config.reference = 0.0;
            }
            SECTION("non-finite reference") {
                config.reference = std::numeric_limits<F64>::infinity();
            }
            SECTION("non-positive epsilon") {
                config.epsilon = 0.0;
            }
            SECTION("non-positive minimum gain") {
                config.minGain = 0.0;
            }
            SECTION("maximum gain below minimum gain") {
                config.maxGain = config.minGain / 2.0;
            }
            SECTION("non-finite maximum gain") {
                config.maxGain = std::numeric_limits<F64>::quiet_NaN();
            }
            SECTION("maximum gain change below one") {
                config.maxGainChange = 0.5;
            }
            SECTION("non-finite maximum gain change") {
                config.maxGainChange =
                    std::numeric_limits<F64>::quiet_NaN();
            }

            RequireAgcValidationError(impl, config);
        }
    }
}

TEST_CASE("AGC - Rejects unsupported dtype", "[modules][agc][validation]") {
    const auto implementations = Registry::ListAvailableModules("agc");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            RequireAgcValidationError(impl, Modules::Agc{}, DataType::U8);
        }
    }
}
