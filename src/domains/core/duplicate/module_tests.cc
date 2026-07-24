#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>

#include "jetstream/domains/core/duplicate/module.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/logger.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

namespace {

TensorMap MakeCpuDuplicateInput(Tensor& input) {
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {2}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = input;
    return inputs;
}

std::shared_ptr<Module> BuildCpuDuplicateModule() {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("duplicate",
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  "generic",
                                  module) == Result::SUCCESS);
    return module;
}

void RequireCpuDuplicateValidationError(const Modules::Duplicate& config) {
    Tensor input;
    const auto inputs = MakeCpuDuplicateInput(input);
    const auto module = BuildCpuDuplicateModule();

    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->outputs().empty());
    REQUIRE(module->outputs().empty());
    REQUIRE(JST_LOG_LAST_ERROR().find("[DUPLICATE]") != std::string::npos);
}

template<typename T>
void expectDuplicateSuccess(const std::string& tag,
                            const std::vector<T>& values) {
    const auto implementations = Registry::ListAvailableModules("duplicate");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Type: " << tag << " Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("duplicate", impl.device, impl.runtime, impl.provider);

            Modules::Duplicate config;
            config.hostAccessible = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<T>({values.size()});
            for (U64 i = 0; i < values.size(); ++i) {
                input.at(i) = values[i];
            }

            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");
            REQUIRE(out.shape(0) == values.size());
            REQUIRE(out.dtype() == TypeToDataType<T>());

            for (U64 i = 0; i < values.size(); ++i) {
                REQUIRE(out.at<T>(i) == values[i]);
            }
        }
    }
}

}  // namespace

TEST_CASE("Duplicate Module - F32", "[modules][duplicate][F32]") {
    expectDuplicateSuccess<F32>("F32", {1.0f, -2.0f, 3.0f, -4.0f});
}

TEST_CASE("Duplicate Module - Full dtype coverage", "[modules][duplicate][all]") {
    expectDuplicateSuccess<F64>("F64", {1.0, -2.0, 3.0});
    expectDuplicateSuccess<I8>("I8", {1, -2, 3, -4});
    expectDuplicateSuccess<I16>("I16", {1024, -2048, 4096});
    expectDuplicateSuccess<I32>("I32", {1024, -2048, 4096});
    expectDuplicateSuccess<I64>("I64", {1024, -2048, 4096});
    expectDuplicateSuccess<U8>("U8", {1, 2, 3, 4});
    expectDuplicateSuccess<U16>("U16", {1024, 2048, 4096});
    expectDuplicateSuccess<U32>("U32", {1024, 2048, 4096});
    expectDuplicateSuccess<U64>("U64", {1024, 2048, 4096});
    expectDuplicateSuccess<CF32>("CF32", {{1.0f, 2.0f}, {3.0f, -4.0f}});
    expectDuplicateSuccess<CF64>("CF64", {{1.0, 2.0}, {3.0, -4.0}});
    expectDuplicateSuccess<CI8>("CI8", {{1, 2}, {3, -4}});
    expectDuplicateSuccess<CI16>("CI16", {{1024, -2048}, {4096, 8192}});
    expectDuplicateSuccess<CI32>("CI32", {{1024, -2048}, {4096, 8192}});
    expectDuplicateSuccess<CI64>("CI64", {{1024, -2048}, {4096, 8192}});
    expectDuplicateSuccess<CU8>("CU8", {{1, 2}, {3, 4}});
    expectDuplicateSuccess<CU16>("CU16", {{1024, 2048}, {4096, 8192}});
    expectDuplicateSuccess<CU32>("CU32", {{1024, 2048}, {4096, 8192}});
    expectDuplicateSuccess<CU64>("CU64", {{1024, 2048}, {4096, 8192}});
}

TEST_CASE("Duplicate Module - Validation contract",
          "[modules][duplicate][validation]") {
    SECTION("empty output device is rejected") {
        Modules::Duplicate config;
        config.outputDevice.clear();
        RequireCpuDuplicateValidationError(config);
    }

    SECTION("invalid spelling stops before define and create") {
        Modules::Duplicate config;
        config.outputDevice = "invalid";
        RequireCpuDuplicateValidationError(config);
    }

    SECTION("WebGPU has no tensor buffer backend") {
        Modules::Duplicate config;
        config.outputDevice = "webgpu";
        RequireCpuDuplicateValidationError(config);
    }

#ifndef JETSTREAM_BACKEND_CUDA_AVAILABLE
    SECTION("uncompiled CUDA target is rejected") {
        Modules::Duplicate config;
        config.outputDevice = "cuda";
        RequireCpuDuplicateValidationError(config);
    }
#else
    SECTION("CPU input requires host-accessible CUDA output") {
        Modules::Duplicate config;
        config.outputDevice = "cuda";
        config.hostAccessible = false;
        RequireCpuDuplicateValidationError(config);
    }
#endif

#ifndef JETSTREAM_BACKEND_METAL_AVAILABLE
    SECTION("uncompiled Metal target is rejected") {
        Modules::Duplicate config;
        config.outputDevice = "metal";
        RequireCpuDuplicateValidationError(config);
    }
#endif

#ifndef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    SECTION("uncompiled Vulkan target is rejected") {
        Modules::Duplicate config;
        config.outputDevice = "vulkan";
        RequireCpuDuplicateValidationError(config);
    }
#else
    SECTION("CPU input follows the selected Vulkan memory model") {
        const auto& vulkan = Backend::State<DeviceType::Vulkan>();
        REQUIRE(vulkan != nullptr);

        Tensor input;
        const auto inputs = MakeCpuDuplicateInput(input);
        const auto module = BuildCpuDuplicateModule();

        Modules::Duplicate config;
        REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);
        const auto outputId = module->outputs().at("buffer").tensor.id();

        Parser::Map candidate;
        candidate["hostAccessible"] = false;
        candidate["outputDevice"] = std::string("vulkan");

        const auto result = module->reconfigure(candidate, true);
        if (!vulkan->isAvailable() || vulkan->hasUnifiedMemory()) {
            REQUIRE(result == Result::SUCCESS);
        } else {
            REQUIRE(result == Result::ERROR);
        }

        REQUIRE(module->state() == Module::State::CREATED);
        REQUIRE(module->outputs().at("buffer").tensor.id() == outputId);
        REQUIRE(module->destroy() == Result::SUCCESS);
    }
#endif
}

TEST_CASE("Duplicate Module - None accepts a non-host-accessible CPU output",
          "[modules][duplicate][validation][output-device]") {
    Tensor input;
    const auto inputs = MakeCpuDuplicateInput(input);
    const auto module = BuildCpuDuplicateModule();

    Modules::Duplicate config;
    config.outputDevice = "NoNe";
    config.hostAccessible = false;

    REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);
    REQUIRE(module->state() == Module::State::CREATED);
    REQUIRE(module->outputs().at("buffer").tensor.device() == DeviceType::CPU);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Duplicate Module - Rejected target preserves live output",
          "[modules][duplicate][validation][reconfigure]") {
    Tensor input;
    const auto inputs = MakeCpuDuplicateInput(input);
    input.at<F32>(0) = 1.0f;
    input.at<F32>(1) = 2.0f;

    const auto module = BuildCpuDuplicateModule();
    Modules::Duplicate config;
    config.outputDevice = "none";
    config.hostAccessible = false;
    REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

    Runtime runtime("test", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);

    const auto outputId = module->outputs().at("buffer").tensor.id();
    Parser::Map rejected;
    rejected["hostAccessible"] = true;
    rejected["outputDevice"] = std::string("webgpu");
    REQUIRE(module->reconfigure(rejected) == Result::ERROR);
    REQUIRE(module->state() == Module::State::CREATED);
    REQUIRE(module->outputs().at("buffer").tensor.id() == outputId);

    const auto& applied = static_cast<const Modules::Duplicate&>(module->config());
    REQUIRE(applied.outputDevice == config.outputDevice);
    REQUIRE(applied.hostAccessible == config.hostAccessible);

    input.at<F32>(0) = 3.0f;
    input.at<F32>(1) = 4.0f;
    skippedModules.clear();
    failedModules.clear();
    REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);

    const auto& output = module->outputs().at("buffer").tensor;
    REQUIRE(output.id() == outputId);
    REQUIRE(output.at<F32>(0) == 3.0f);
    REQUIRE(output.at<F32>(1) == 4.0f);

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Duplicate Module - None preserves input device",
          "[modules][duplicate][output-device]") {
    const auto implementations = Registry::ListAvailableModules("duplicate");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("duplicate",
                                          impl.device,
                                          impl.runtime,
                                          impl.provider,
                                          module) == Result::SUCCESS);

            TypedTensor<I32> cpuInput(DeviceType::CPU, {2});
            cpuInput.at(0) = 10;
            cpuInput.at(1) = 20;

            Tensor input = cpuInput;
            if (impl.device != DeviceType::CPU) {
                input = Tensor(impl.device, cpuInput);
            }

            Modules::Duplicate config;
            config.outputDevice = "NoNe";

            TensorMap inputs;
            inputs["buffer"].requested("test", "buffer");
            inputs["buffer"].tensor = input;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("test", impl.device, impl.runtime);
            REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);
            REQUIRE(skippedModules.empty());

            const auto& output = module->outputs().at("buffer").tensor;
            REQUIRE(output.device() == impl.device);

            Tensor cpuOutput = output;
            if (output.device() != DeviceType::CPU) {
                cpuOutput = Tensor(DeviceType::CPU, output);
            }
            REQUIRE(cpuOutput.at<I32>(0) == 10);
            REQUIRE(cpuOutput.at<I32>(1) == 20);

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Duplicate Module - CUDA preserves non-contiguous views", "[modules][duplicate][CUDA][view]") {
    const auto implementations = Registry::ListAvailableModules("duplicate", DeviceType::CUDA);
    if (implementations.empty()) {
        SUCCEED("CUDA duplicate module is unavailable in this build.");
        return;
    }

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("duplicate",
                                          impl.device,
                                          impl.runtime,
                                          impl.provider,
                                          module) == Result::SUCCESS);

            TypedTensor<I32> cpuInput(DeviceType::CPU, {2, 3, 2, 2});
            I32 value = 1;
            for (U64 a = 0; a < cpuInput.shape(0); ++a) {
                for (U64 b = 0; b < cpuInput.shape(1); ++b) {
                    for (U64 c = 0; c < cpuInput.shape(2); ++c) {
                        for (U64 d = 0; d < cpuInput.shape(3); ++d) {
                            cpuInput.at(a, b, c, d) = value++;
                        }
                    }
                }
            }

            Tensor deviceInput(impl.device, cpuInput);
            Tensor view = deviceInput.clone();
            REQUIRE(view.permute({2, 0, 3, 1}) == Result::SUCCESS);
            REQUIRE_FALSE(view.contiguous());

            Modules::Duplicate config;
            config.hostAccessible = true;

            TensorMap inputs;
            inputs["buffer"].requested("test", "buffer");
            inputs["buffer"].tensor = view;

            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("test", impl.device, impl.runtime);
            REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);
            REQUIRE(skippedModules.empty());

            const auto& output = module->outputs().at("buffer").tensor;
            REQUIRE(output.shape() == Shape{2, 2, 2, 3});
            REQUIRE(output.contiguous());

            Tensor cpuOutput = output;
            if (cpuOutput.device() != DeviceType::CPU) {
                cpuOutput = Tensor(DeviceType::CPU, output);
            }

            for (U64 a = 0; a < cpuOutput.shape(0); ++a) {
                for (U64 b = 0; b < cpuOutput.shape(1); ++b) {
                    for (U64 c = 0; c < cpuOutput.shape(2); ++c) {
                        for (U64 d = 0; d < cpuOutput.shape(3); ++d) {
                            REQUIRE(cpuOutput.at<I32>(a, b, c, d) ==
                                    cpuInput.at(b, d, a, c));
                        }
                    }
                }
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Duplicate Module - CPU runtime can target CUDA output", "[modules][duplicate][CPU][output-device]") {
    const auto cpuImplementations = Registry::ListAvailableModules("duplicate", DeviceType::CPU);
    const auto cudaImplementations = Registry::ListAvailableModules("duplicate", DeviceType::CUDA);
    if (cpuImplementations.empty() || cudaImplementations.empty()) {
        SUCCEED("CPU or CUDA duplicate module is unavailable in this build.");
        return;
    }

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("duplicate",
                                  DeviceType::CPU,
                                  RuntimeType::NATIVE,
                                  "generic",
                                  module) == Result::SUCCESS);

    TypedTensor<F32> input(DeviceType::CPU, {4});
    input.at(0) = 1.0f;
    input.at(1) = -2.0f;
    input.at(2) = 3.0f;
    input.at(3) = -4.0f;

    Modules::Duplicate config;
    config.hostAccessible = true;
    config.outputDevice = GetDeviceName(DeviceType::CUDA);

    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = input;

    REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

    Runtime runtime("test", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.empty());

    const auto& output = module->outputs().at("buffer").tensor;
    REQUIRE(output.device() == DeviceType::CUDA);
    REQUIRE(output.shape() == Shape{4});

    Tensor cpuOutput(DeviceType::CPU, output);
    REQUIRE(cpuOutput.at<F32>(0) == 1.0f);
    REQUIRE(cpuOutput.at<F32>(1) == -2.0f);
    REQUIRE(cpuOutput.at<F32>(2) == 3.0f);
    REQUIRE(cpuOutput.at<F32>(3) == -4.0f);

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Duplicate Module - CUDA runtime can target CPU output", "[modules][duplicate][CUDA][output-device]") {
    const auto implementations = Registry::ListAvailableModules("duplicate", DeviceType::CUDA);
    if (implementations.empty()) {
        SUCCEED("CUDA duplicate module is unavailable in this build.");
        return;
    }

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("duplicate",
                                          impl.device,
                                          impl.runtime,
                                          impl.provider,
                                          module) == Result::SUCCESS);

            TypedTensor<I32> cpuInput(DeviceType::CPU, {4});
            cpuInput.at(0) = 10;
            cpuInput.at(1) = 20;
            cpuInput.at(2) = 30;
            cpuInput.at(3) = 40;

            Tensor deviceInput(impl.device, cpuInput);

            Modules::Duplicate config;
            config.hostAccessible = true;
            config.outputDevice = GetDeviceName(DeviceType::CPU);

            TensorMap inputs;
            inputs["buffer"].requested("test", "buffer");
            inputs["buffer"].tensor = deviceInput;

            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("test", impl.device, impl.runtime);
            REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);
            REQUIRE(skippedModules.empty());

            const auto& output = module->outputs().at("buffer").tensor;
            REQUIRE(output.device() == DeviceType::CPU);
            REQUIRE(output.shape() == Shape{4});
            REQUIRE(output.at<I32>(0) == 10);
            REQUIRE(output.at<I32>(1) == 20);
            REQUIRE(output.at<I32>(2) == 30);
            REQUIRE(output.at<I32>(3) == 40);

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}
