#include <catch2/catch_test_macros.hpp>

#include "dmi_module.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

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
