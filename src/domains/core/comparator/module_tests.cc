#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/comparator/module.hh"

using namespace Jetstream;

namespace {

TensorMap MakeComparatorInputs(const Registry::ModuleRegistration& impl,
                               const std::vector<std::pair<DataType, Shape>>& specifications) {
    TensorMap inputs;
    for (U64 i = 0; i < specifications.size(); ++i) {
        Tensor tensor;
        REQUIRE(tensor.create(impl.device,
                              specifications[i].first,
                              specifications[i].second) == Result::SUCCESS);

        const auto port = "input" + std::to_string(i);
        inputs[port].requested("test", port);
        inputs[port].tensor = std::move(tensor);
    }
    return inputs;
}

void RequireComparatorValidationError(const Registry::ModuleRegistration& impl,
                                      const Modules::Comparator& config,
                                      TensorMap inputs) {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("comparator", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

TEST_CASE("Comparator Module - F32 Equal", "[modules][comparator][F32]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("comparator", impl.device, impl.runtime, impl.provider);

            Modules::Comparator config;
            config.inputCount = 2;
            config.tolerance = 1e-6;
            ctx.setConfig(config);

            auto a = ctx.createTensor<F32>({4});
            auto b = ctx.createTensor<F32>({4});

            a.at(0) = 1.0f; a.at(1) = 2.0f; a.at(2) = 3.0f; a.at(3) = 4.0f;
            b.at(0) = 1.0f; b.at(1) = 2.0f; b.at(2) = 3.0f; b.at(3) = 4.0f;

            ctx.setInput("input0", a);
            ctx.setInput("input1", b);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("error");
            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        }
    }
}

TEST_CASE("Comparator Module - F32 Unequal", "[modules][comparator][F32]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("comparator", impl.device, impl.runtime, impl.provider);

            Modules::Comparator config;
            config.inputCount = 2;
            config.tolerance = 1e-6;
            ctx.setConfig(config);

            auto a = ctx.createTensor<F32>({4});
            auto b = ctx.createTensor<F32>({4});

            a.at(0) = 1.0f; a.at(1) = 2.0f; a.at(2) = 3.0f; a.at(3) = 4.0f;
            b.at(0) = 1.0f; b.at(1) = 2.5f; b.at(2) = 3.0f; b.at(3) = 5.0f;

            ctx.setInput("input0", a);
            ctx.setInput("input1", b);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("error");
            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(0.5f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
        }
    }
}

TEST_CASE("Comparator Module - CF32", "[modules][comparator][CF32]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("comparator", impl.device, impl.runtime, impl.provider);

            Modules::Comparator config;
            config.inputCount = 2;
            ctx.setConfig(config);

            auto a = ctx.createTensor<CF32>({2});
            auto b = ctx.createTensor<CF32>({2});

            a.at(0) = {3.0f, 4.0f};
            a.at(1) = {0.0f, 0.0f};
            b.at(0) = {0.0f, 0.0f};
            b.at(1) = {0.0f, 0.0f};

            ctx.setInput("input0", a);
            ctx.setInput("input1", b);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("error");
            REQUIRE(out.dtype() == DataType::F32);
            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(5.0f, 1e-5f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        }
    }
}

TEST_CASE("Comparator Module - F64", "[modules][comparator][F64]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("comparator", impl.device, impl.runtime, impl.provider);

            Modules::Comparator config;
            config.inputCount = 2;
            ctx.setConfig(config);

            auto a = ctx.createTensor<F64>({2});
            auto b = ctx.createTensor<F64>({2});

            a.at(0) = 1.0; a.at(1) = 2.0;
            b.at(0) = 1.5; b.at(1) = 2.0;

            ctx.setInput("input0", a);
            ctx.setInput("input1", b);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("error");
            REQUIRE(out.dtype() == DataType::F64);
            REQUIRE_THAT(out.at<F64>(0), Catch::Matchers::WithinAbs(0.5, 1e-12));
            REQUIRE_THAT(out.at<F64>(1), Catch::Matchers::WithinAbs(0.0, 1e-12));
        }
    }
}

TEST_CASE("Comparator Module - CF64", "[modules][comparator][CF64]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("comparator", impl.device, impl.runtime, impl.provider);

            Modules::Comparator config;
            config.inputCount = 2;
            ctx.setConfig(config);

            auto a = ctx.createTensor<CF64>({1});
            auto b = ctx.createTensor<CF64>({1});

            a.at(0) = {0.0, 0.0};
            b.at(0) = {3.0, 4.0};

            ctx.setInput("input0", a);
            ctx.setInput("input1", b);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("error");
            REQUIRE(out.dtype() == DataType::F64);
            REQUIRE_THAT(out.at<F64>(0), Catch::Matchers::WithinAbs(5.0, 1e-12));
        }
    }
}

TEST_CASE("Comparator Module - Three Inputs", "[modules][comparator][multi]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("comparator", impl.device, impl.runtime, impl.provider);

            Modules::Comparator config;
            config.inputCount = 3;
            ctx.setConfig(config);

            auto a = ctx.createTensor<F32>({2});
            auto b = ctx.createTensor<F32>({2});
            auto c = ctx.createTensor<F32>({2});

            a.at(0) = 0.0f; a.at(1) = 0.0f;
            b.at(0) = 1.0f; b.at(1) = 0.0f;
            c.at(0) = 0.0f; c.at(1) = 2.0f;

            ctx.setInput("input0", a);
            ctx.setInput("input1", b);
            ctx.setInput("input2", c);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("error");
            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
        }
    }
}

TEST_CASE("Comparator Module - Validation contract",
          "[modules][comparator][validation]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("candidate count is validated before topology") {
                Modules::Comparator config;
                config.inputCount = 1;
                RequireComparatorValidationError(impl, config, {});

                config.inputCount = 17;
                RequireComparatorValidationError(impl, config, {});
            }

            SECTION("candidate tolerance is validated before allocation") {
                Modules::Comparator config;
                auto inputs = MakeComparatorInputs(impl, {
                    {DataType::F32, Shape{4}},
                    {DataType::F32, Shape{4}},
                });

                config.tolerance = -1.0;
                RequireComparatorValidationError(impl, config, inputs);

                config.tolerance = std::numeric_limits<F64>::quiet_NaN();
                RequireComparatorValidationError(impl, config, inputs);

                config.tolerance = std::numeric_limits<F64>::infinity();
                RequireComparatorValidationError(impl, config, std::move(inputs));
            }

            SECTION("input count must exactly match the candidate topology") {
                Modules::Comparator config;
                config.inputCount = 3;
                auto inputs = MakeComparatorInputs(impl, {
                    {DataType::F32, Shape{4}},
                    {DataType::F32, Shape{4}},
                });
                RequireComparatorValidationError(impl, config, std::move(inputs));
            }

            SECTION("undeclared direct inputs are rejected") {
                Modules::Comparator config;
                auto inputs = MakeComparatorInputs(impl, {
                    {DataType::F32, Shape{4}},
                    {DataType::F32, Shape{4}},
                });
                Tensor extra;
                REQUIRE(extra.create(impl.device, DataType::F32, {4}) == Result::SUCCESS);
                inputs["extra"].requested("test", "extra");
                inputs["extra"].tensor = std::move(extra);
                RequireComparatorValidationError(impl, config, std::move(inputs));
            }

            SECTION("input shapes must match exactly") {
                Modules::Comparator config;
                auto inputs = MakeComparatorInputs(impl, {
                    {DataType::F32, Shape{4}},
                    {DataType::F32, Shape{3}},
                });
                RequireComparatorValidationError(impl, config, std::move(inputs));
            }

            SECTION("input dtypes must match exactly") {
                Modules::Comparator config;
                auto inputs = MakeComparatorInputs(impl, {
                    {DataType::F32, Shape{4}},
                    {DataType::F64, Shape{4}},
                });
                RequireComparatorValidationError(impl, config, std::move(inputs));
            }

            if (impl.device == DeviceType::CPU) {
                SECTION("CPU provider rejects unsupported dtypes before allocation") {
                    Modules::Comparator config;
                    auto inputs = MakeComparatorInputs(impl, {
                        {DataType::U8, Shape{4}},
                        {DataType::U8, Shape{4}},
                    });
                    RequireComparatorValidationError(impl, config, std::move(inputs));
                }

                SECTION("CPU provider rejects unsupported allocation sizes") {
                    const U64 size = std::numeric_limits<U64>::max() / sizeof(F32);
                    TensorMap inputs;
                    for (U64 i = 0; i < 2; ++i) {
                        Tensor tensor;
                        REQUIRE(tensor.create(impl.device, DataType::F32, {1}) ==
                                Result::SUCCESS);
                        REQUIRE(tensor.broadcastTo({size}) == Result::SUCCESS);
                        const auto port = "input" + std::to_string(i);
                        inputs[port].requested("test", port);
                        inputs[port].tensor = std::move(tensor);
                    }
                    RequireComparatorValidationError(impl, Modules::Comparator{},
                                                     std::move(inputs));
                }
            }

            SECTION("rank-zero inputs are rejected before create") {
                auto inputs = MakeComparatorInputs(impl, {
                    {DataType::F32, Shape{1}},
                    {DataType::F32, Shape{1}},
                });
                REQUIRE(inputs.at("input0").tensor.squeezeDims(0) == Result::SUCCESS);
                REQUIRE(inputs.at("input1").tensor.squeezeDims(0) == Result::SUCCESS);
                RequireComparatorValidationError(impl, Modules::Comparator{},
                                                 std::move(inputs));
            }
        }
    }
}

TEST_CASE("Comparator Module - Count changes retain RECREATE semantics",
          "[modules][comparator][validation][reconfigure]") {
    const auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            auto inputs = MakeComparatorInputs(impl, {
                {DataType::F32, Shape{4}},
                {DataType::F32, Shape{4}},
            });
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("comparator", impl.device, impl.runtime,
                                          impl.provider, module) == Result::SUCCESS);
            REQUIRE(module->create("test", Modules::Comparator{}, inputs) == Result::SUCCESS);

            Parser::Map update;
            update["inputCount"] = U64{3};
            REQUIRE(module->reconfigure(update) == Result::RECREATE);
            REQUIRE(module->state() == Module::State::CREATED);
            REQUIRE(static_cast<const Modules::Comparator&>(module->config()).inputCount == 2);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Comparator Module - Validation-only and rejected updates preserve live compute",
          "[modules][comparator][validation][reconfigure]") {
    auto implementations = Registry::ListAvailableModules("comparator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            auto inputs = MakeComparatorInputs(impl, {
                {DataType::F32, Shape{1}},
                {DataType::F32, Shape{1}},
            });
            inputs.at("input0").tensor.at<F32>(0) = 0.0f;
            inputs.at("input1").tensor.at<F32>(0) = 0.25f;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("comparator", impl.device, impl.runtime,
                                          impl.provider, module) == Result::SUCCESS);
            Modules::Comparator config;
            config.tolerance = 0.5;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("test", impl.device, impl.runtime);
            REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);

            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);
            REQUIRE_THAT(module->outputs().at("error").tensor.at<F32>(0),
                         Catch::Matchers::WithinAbs(0.25f, 1e-6f));

            const auto outputId = module->outputs().at("error").tensor.id();

            Parser::Map validateOnly;
            validateOnly["tolerance"] = F64{0.1};
            REQUIRE(module->reconfigure(validateOnly, true) == Result::SUCCESS);

            inputs.at("input1").tensor.at<F32>(0) = 0.4f;
            skippedModules.clear();
            failedModules.clear();
            REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);
            REQUIRE(module->outputs().at("error").tensor.id() == outputId);
            REQUIRE_THAT(module->outputs().at("error").tensor.at<F32>(0),
                         Catch::Matchers::WithinAbs(0.4f, 1e-6f));

            Parser::Map rejected;
            rejected["tolerance"] = F64{-1.0};
            REQUIRE(module->reconfigure(rejected) == Result::ERROR);
            REQUIRE(module->state() == Module::State::CREATED);
            REQUIRE(static_cast<const Modules::Comparator&>(module->config()).tolerance ==
                    config.tolerance);

            inputs.at("input1").tensor.at<F32>(0) = 0.75f;
            skippedModules.clear();
            failedModules.clear();
            REQUIRE(runtime.compute({}, skippedModules, failedModules) == Result::SUCCESS);
            REQUIRE(module->outputs().at("error").tensor.id() == outputId);
            REQUIRE_THAT(module->outputs().at("error").tensor.at<F32>(0),
                         Catch::Matchers::WithinAbs(0.75f, 1e-6f));

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}
