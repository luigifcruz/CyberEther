#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <type_traits>
#include <unordered_set>

#include "jetstream/domains/core/ones_tensor/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

namespace {

template<typename T>
void ExpectOnesTensorSuccess(const std::vector<U64>& shape, const T& expected) {
    const auto implementations = Registry::ListAvailableModules("ones_tensor");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Type: " << NumericTypeInfo<T>::name << " Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("ones_tensor", impl.device, impl.runtime, impl.provider);

            Modules::OnesTensor config;
            config.shape = shape;
            config.dataType = NumericTypeInfo<T>::name;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");
            REQUIRE(out.shape() == shape);
            REQUIRE(out.dtype() == TypeToDataType<T>());
            REQUIRE(out.size() > 0);

            const T* outData = out.data<T>();
            for (U64 i = 0; i < out.size(); ++i) {
                if constexpr (std::is_same_v<T, F32>) {
                    REQUIRE_THAT(outData[i], Catch::Matchers::WithinAbs(expected, 1e-6f));
                } else {
                    REQUIRE_THAT(outData[i].real(), Catch::Matchers::WithinAbs(expected.real(), 1e-6f));
                    REQUIRE_THAT(outData[i].imag(), Catch::Matchers::WithinAbs(expected.imag(), 1e-6f));
                }
            }
        }
    }
}

}  // namespace

TEST_CASE("Ones Tensor Module - F32", "[modules][ones_tensor][F32]") {
    ExpectOnesTensorSuccess<F32>({2, 3}, 1.0f);
}

TEST_CASE("Ones Tensor Module - CF32", "[modules][ones_tensor][CF32]") {
    ExpectOnesTensorSuccess<CF32>({2, 2, 2}, CF32(1.0f, 0.0f));
}

TEST_CASE("Ones Tensor Module - Repeated compute keeps the same tensor",
          "[modules][ones_tensor][continuity]") {
    const auto implementations = Registry::ListAvailableModules("ones_tensor");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("ones_tensor",
                                          impl.device,
                                          impl.runtime,
                                          impl.provider,
                                          module) == Result::SUCCESS);

            Modules::OnesTensor config;
            config.shape = {4};
            config.dataType = "F32";

            REQUIRE(module->create("test", config, TensorMap{}) == Result::SUCCESS);

            Runtime runtime("test", impl.device, impl.runtime);
            REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skippedModules;
            REQUIRE(runtime.compute({}, skippedModules) == Result::SUCCESS);
            REQUIRE(skippedModules.empty());

            const auto initialId = module->outputs().at("buffer").tensor.id();
            Tensor firstOutput = module->outputs().at("buffer").tensor;
            if (firstOutput.device() != DeviceType::CPU) {
                firstOutput = Tensor(DeviceType::CPU, firstOutput);
            }
            REQUIRE_THAT(firstOutput.at<F32>(0), Catch::Matchers::WithinAbs(1.0f, 1e-6f));

            skippedModules.clear();
            REQUIRE(runtime.compute({}, skippedModules) == Result::SUCCESS);
            REQUIRE(skippedModules.empty());

            const auto& secondOutput = module->outputs().at("buffer").tensor;
            REQUIRE(secondOutput.id() == initialId);

            Tensor secondCpuOutput = secondOutput;
            if (secondCpuOutput.device() != DeviceType::CPU) {
                secondCpuOutput = Tensor(DeviceType::CPU, secondCpuOutput);
            }
            REQUIRE_THAT(secondCpuOutput.at<F32>(0), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(secondCpuOutput.at<F32>(3), Catch::Matchers::WithinAbs(1.0f, 1e-6f));

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Ones Tensor Module - Validation rejects invalid config",
          "[modules][ones_tensor][validation]") {
    const auto implementations = Registry::ListAvailableModules("ones_tensor");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        SECTION("empty shape") {
            TestContext ctx("ones_tensor", impl.device, impl.runtime, impl.provider);

            Modules::OnesTensor config;
            config.shape = {};
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("zero dimension") {
            TestContext ctx("ones_tensor", impl.device, impl.runtime, impl.provider);

            Modules::OnesTensor config;
            config.shape = {2, 0, 4};
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("invalid dtype") {
            TestContext ctx("ones_tensor", impl.device, impl.runtime, impl.provider);

            Modules::OnesTensor config;
            config.dataType = "I32";
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
