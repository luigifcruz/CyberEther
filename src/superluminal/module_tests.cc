#include <catch2/catch_test_macros.hpp>

#include "dmi_module.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/registry.hh"
#include "jetstream/superluminal.hh"

using namespace Jetstream;

namespace Jetstream {

JETSTREAM_API Result PrepareSuperluminalPlotBuffer(const std::string& name,
                                                   const Superluminal::PlotConfig& config,
                                                   Superluminal::PlotConfig& resolvedConfig);

}  // namespace Jetstream

TEST_CASE("Superluminal plot buffers receive signal axes",
          "[superluminal][signal_axes]") {
    Superluminal::PlotConfig config;
    REQUIRE(config.buffer.create(DeviceType::CPU, DataType::F32, {1, 8192}) ==
            Result::SUCCESS);

    Superluminal::PlotConfig resolved;
    REQUIRE(PrepareSuperluminalPlotBuffer("example", config, resolved) ==
            Result::SUCCESS);

    SignalAxes axes;
    REQUIRE(ResolveSignalAxes(resolved.buffer, axes) == Result::SUCCESS);
    REQUIRE(axes.sample == Index{1});
    REQUIRE(axes.batch == Index{0});
    REQUIRE_FALSE(axes.channel);

    REQUIRE_FALSE(config.buffer.hasAttribute(std::string(SampleAxisAttribute)));
    REQUIRE_FALSE(config.buffer.hasAttribute(std::string(BatchAxisAttribute)));
}

TEST_CASE("Superluminal plot buffers honor an explicit channel axis",
          "[superluminal][signal_axes]") {
    Superluminal::PlotConfig config;
    REQUIRE(config.buffer.create(DeviceType::CPU, DataType::CF32, {42, 8192}) ==
            Result::SUCCESS);
    config.channelAxis = 0;
    config.channelIndex = 3;

    Superluminal::PlotConfig resolved;
    REQUIRE(PrepareSuperluminalPlotBuffer("channels", config, resolved) ==
            Result::SUCCESS);

    SignalAxes axes;
    REQUIRE(ResolveSignalAxes(resolved.buffer, axes) == Result::SUCCESS);
    REQUIRE(axes.sample == Index{1});
    REQUIRE_FALSE(axes.batch);
    REQUIRE(axes.channel == Index{0});
}

#if defined(JETSTREAM_BACKEND_CUDA_AVAILABLE)
constexpr DeviceType kCrossDevice = DeviceType::CUDA;
#elif defined(JETSTREAM_BACKEND_METAL_AVAILABLE)
constexpr DeviceType kCrossDevice = DeviceType::Metal;
#elif defined(JETSTREAM_BACKEND_VULKAN_AVAILABLE)
constexpr DeviceType kCrossDevice = DeviceType::Vulkan;
#endif

TEST_CASE("Dynamic Tensor Import Module - Validation rejects an uninitialized tensor",
          "[modules][dynamic_tensor_import][validation]") {
    const auto implementations =
        Registry::ListAvailableModules("dynamic_tensor_import");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("dynamic_tensor_import",
                                          impl.device,
                                          impl.runtime,
                                          impl.provider,
                                          module) == Result::SUCCESS);

            Modules::DynamicTensorImport config;
            REQUIRE(module->create("test", config, {}) == Result::ERROR);
            REQUIRE(module->state() == Module::State::ERRORED);
            REQUIRE(module->interface()->outputs().empty());
            REQUIRE(module->outputs().empty());
        }
    }
}

TEST_CASE("Dynamic Tensor Import Module - Imports an initialized tensor",
          "[modules][dynamic_tensor_import]") {
    const auto implementations =
        Registry::ListAvailableModules("dynamic_tensor_import");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Tensor buffer;
            REQUIRE(buffer.create(DeviceType::CPU, DataType::F32, {4}) ==
                    Result::SUCCESS);
            buffer.at<F32>(0) = 3.0f;

            Modules::DynamicTensorImport config;
            config.buffer = buffer;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("dynamic_tensor_import",
                                          impl.device,
                                          impl.runtime,
                                          impl.provider,
                                          module) == Result::SUCCESS);
            REQUIRE(module->create("test", config, {}) == Result::SUCCESS);

            const auto& output = module->outputs().at("buffer").tensor;
            REQUIRE(output.id() == buffer.id());
            REQUIRE(output.data<F32>() == buffer.data<F32>());
            REQUIRE(output.at<F32>(0) == 3.0f);

            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Dynamic Tensor Import Module - Accepts an initialized zero-extent tensor",
          "[modules][dynamic_tensor_import][zero_extent]") {
    const auto implementations =
        Registry::ListAvailableModules("dynamic_tensor_import");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Tensor buffer;
            REQUIRE(buffer.create(DeviceType::CPU, DataType::F32, {0}) ==
                    Result::SUCCESS);

            Modules::DynamicTensorImport config;
            config.buffer = buffer;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("dynamic_tensor_import",
                                          impl.device,
                                          impl.runtime,
                                          impl.provider,
                                          module) == Result::SUCCESS);
            REQUIRE(module->create("test", config, {}) == Result::SUCCESS);

            const auto& output = module->outputs().at("buffer").tensor;
            REQUIRE(output.id() == buffer.id());
            REQUIRE(output.shape() == Shape{0});
            REQUIRE(output.size() == 0);

            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

#if defined(JETSTREAM_BACKEND_CUDA_AVAILABLE) || \
    defined(JETSTREAM_BACKEND_METAL_AVAILABLE) || \
    defined(JETSTREAM_BACKEND_VULKAN_AVAILABLE)
TEST_CASE("Dynamic Tensor Import Module - Accepts a cross-device tensor",
          "[modules][dynamic_tensor_import][cross_device]") {
    const auto implementations =
        Registry::ListAvailableModules("dynamic_tensor_import");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            REQUIRE(impl.device != kCrossDevice);

            Tensor buffer;
            REQUIRE(buffer.create(kCrossDevice, DataType::F32, {0}) ==
                    Result::SUCCESS);

            Modules::DynamicTensorImport config;
            config.buffer = buffer;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("dynamic_tensor_import",
                                          impl.device,
                                          impl.runtime,
                                          impl.provider,
                                          module) == Result::SUCCESS);
            REQUIRE(module->create("test", config, {}) == Result::SUCCESS);

            const auto& output = module->outputs().at("buffer").tensor;
            REQUIRE(output.id() == buffer.id());
            REQUIRE(output.device() == kCrossDevice);

            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}
#endif
