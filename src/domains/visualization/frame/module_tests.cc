#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <limits>

#include "jetstream/domains/visualization/frame/module.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

namespace {

void RequireFrameValidationError(const Registry::ModuleRegistration& impl,
                                 const DataType dtype,
                                 const Shape& shape,
                                 const bool broadcast = false) {
    Tensor input;
    if (broadcast) {
        REQUIRE(input.create(impl.device, dtype, Shape(shape.size(), 1)) == Result::SUCCESS);
        REQUIRE(input.broadcastTo(shape) == Result::SUCCESS);
    } else {
        REQUIRE(input.create(impl.device, dtype, shape) == Result::SUCCESS);
    }

    TensorMap inputs;
    inputs["frame"].requested("test", "frame");
    inputs["frame"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("frame", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", Modules::Frame{}, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs().empty());
}

}  // namespace

TEST_CASE("Frame module accepts valid F32 frames", "[modules][frame]") {
    auto implementations = Registry::ListAvailableModules("frame");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("frame", impl.device, impl.runtime, impl.provider);

            Tensor scalar;
            REQUIRE(scalar.create(DeviceType::CPU, DataType::F32, {16, 32}) ==
                    Result::SUCCESS);
            ctx.setInput("frame", scalar);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Modules::Frame config;
            config.lut = true;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor rgb;
            REQUIRE(rgb.create(DeviceType::CPU, DataType::F32, {16, 32, 3}) ==
                    Result::SUCCESS);
            ctx.setInput("frame", rgb);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor scalarChannel;
            REQUIRE(scalarChannel.create(DeviceType::CPU, DataType::F32, {16, 32, 1}) ==
                    Result::SUCCESS);
            ctx.setInput("frame", scalarChannel);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Tensor rgba;
            REQUIRE(rgba.create(DeviceType::CPU, DataType::F32, {16, 32, 4}) ==
                    Result::SUCCESS);
            ctx.setInput("frame", rgba);
            REQUIRE(ctx.run() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Frame module rejects invalid inputs", "[modules][frame][validation]") {
    auto implementations = Registry::ListAvailableModules("frame");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("dtype must be F32") {
                RequireFrameValidationError(impl, DataType::U8, {16, 32});
            }

            SECTION("rank must be two or three") {
                RequireFrameValidationError(impl, DataType::F32, {32});
                RequireFrameValidationError(impl, DataType::F32, {2, 2, 2, 2});
            }

            SECTION("channels must be one, three, or four") {
                RequireFrameValidationError(impl, DataType::F32, {16, 32, 2});
            }
        }
    }
}

TEST_CASE("Frame module rejects unsupported rendering size",
          "[modules][frame][validation][size]") {
    const auto implementations = Registry::ListAvailableModules("frame");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            const U64 maxElementCount = std::min({
                static_cast<U64>(std::numeric_limits<I32>::max()),
                static_cast<U64>(std::numeric_limits<std::size_t>::max()) / sizeof(F32),
                static_cast<U64>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(F32),
            });
            const Shape shape = {
                1,
                maxElementCount + 1,
            };
            RequireFrameValidationError(impl, DataType::F32, shape, true);
        }
    }
}

TEST_CASE("Frame module supports repeated configurations",
           "[modules][frame][state]") {
    auto implementations = Registry::ListAvailableModules("frame");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("frame", impl.device, impl.runtime, impl.provider);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {8, 8}) ==
                    Result::SUCCESS);
            ctx.setInput("frame", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE(ctx.run() == Result::SUCCESS);

            Modules::Frame config;
            config.lut = true;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);
        }
    }
}
