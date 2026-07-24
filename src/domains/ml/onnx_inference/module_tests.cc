#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <utility>

#include "jetstream/domains/ml/onnx_inference/module.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/platform.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

namespace {

Modules::OnnxInference TestConfig() {
    Modules::OnnxInference config;
    config.modelPath = "missing-jetstream-onnx-model-for-module-test.onnx";
    config.inputNames = {"input"};
    config.outputNames = {"output"};
    return config;
}

TensorMap TestInput(const Registry::ModuleRegistration& impl,
                    const DataType dtype = DataType::F32,
                    const bool rankZero = false) {
    Tensor input;
    const Shape shape = rankZero ? Shape{1} : Shape{4};
    REQUIRE(input.create(impl.device, dtype, shape) == Result::SUCCESS);
    if (rankZero) {
        REQUIRE(input.squeezeDims(0) == Result::SUCCESS);
    }

    TensorMap inputs;
    inputs["input_0"].requested("test", "input_0");
    inputs["input_0"].tensor = std::move(input);
    return inputs;
}

void RequireValidationError(const Registry::ModuleRegistration& impl,
                            const Modules::OnnxInference& config,
                            const DataType dtype = DataType::F32,
                            const bool rankZero = false) {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("onnx_inference", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, TestInput(impl, dtype, rankZero)) ==
            Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs().empty());
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("ONNX inference module validates execution provider spelling before create",
          "[modules][onnx_inference][validation][provider]") {
    const auto implementations = Registry::ListAvailableModules("onnx_inference");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            auto config = TestConfig();
            config.executionProvider = "unknown";
            RequireValidationError(impl, config);
        }
    }
}

TEST_CASE("ONNX inference module leaves provider availability in create",
          "[modules][onnx_inference][validation][provider][external]") {
    const auto implementations = Registry::ListAvailableModules("onnx_inference");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const auto* provider : {"coreml", "tensorrt"}) {
            DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime
                            << " Provider: " << provider) {
                auto config = TestConfig();
                config.executionProvider = provider;

                std::shared_ptr<Module> module;
                REQUIRE(Registry::BuildModule("onnx_inference", impl.device,
                                              impl.runtime, impl.provider, module) ==
                        Result::SUCCESS);
                REQUIRE(module->create("test", config, TestInput(impl)) ==
                        Result::ERROR);
                REQUIRE(module->state() == Module::State::DESTROYED);
                REQUIRE(module->interface()->inputs().size() == 1);
            }
        }
    }
}

TEST_CASE("ONNX inference module validates bound input metadata before create",
          "[modules][onnx_inference][validation][input]") {
    const auto implementations = Registry::ListAvailableModules("onnx_inference");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            const auto config = TestConfig();

            SECTION("unsupported dtype") {
                RequireValidationError(impl, config, DataType::CF32);
            }

            SECTION("rank zero") {
                RequireValidationError(impl, config, DataType::F32, true);
            }
        }
    }
}

TEST_CASE("ONNX inference module leaves model readiness in create",
          "[modules][onnx_inference][validation][external]") {
    const auto implementations = Registry::ListAvailableModules("onnx_inference");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            auto config = TestConfig();
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("onnx_inference", impl.device, impl.runtime,
                                          impl.provider, module) == Result::SUCCESS);

            SECTION("empty path remains incomplete") {
                config.modelPath.clear();
                REQUIRE(module->create("test", config, TestInput(impl)) ==
                        Result::INCOMPLETE);
                REQUIRE(module->state() == Module::State::INCOMPLETE);
                REQUIRE(module->interface()->inputs().size() == 1);
                REQUIRE(module->destroy() == Result::SUCCESS);
            }

            SECTION("missing path remains an external create failure") {
                REQUIRE_FALSE(std::filesystem::exists(
                    Platform::PathFromUtf8(config.modelPath)));
                REQUIRE(module->create("test", config, TestInput(impl)) ==
                        Result::ERROR);
                REQUIRE(module->state() == Module::State::DESTROYED);
                REQUIRE(module->interface()->inputs().size() == 1);
            }
        }
    }
}
