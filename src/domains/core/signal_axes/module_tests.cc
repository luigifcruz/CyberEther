#include <catch2/catch_test_macros.hpp>

#include <any>
#include <memory>
#include <string>

#include "jetstream/domains/core/signal_axes/module.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

namespace {

std::shared_ptr<Module> CreateSignalAxesModule(
    const Registry::ModuleRegistration& implementation,
    const Modules::SignalAxes& config,
    const Tensor& input,
    const Result expected = Result::SUCCESS) {
    TensorMap inputs;
    inputs["buffer"].requested("source", "buffer");
    inputs["buffer"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("signal_axes",
                                  implementation.device,
                                  implementation.runtime,
                                  implementation.provider,
                                  module) == Result::SUCCESS);
    REQUIRE(module->create("signal_axes", config, inputs) == expected);
    return module;
}

}  // namespace

TEST_CASE("Signal axes module replaces roles without copying storage",
          "[modules][signal_axes][metadata]") {
    const auto implementations = Registry::ListAvailableModules("signal_axes");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input(implementation.device, DataType::F32, {2, 3, 4, 5});
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{3},
                .batch = Index{0},
                .channel = Index{2},
            }) == Result::SUCCESS);
            REQUIRE(input.setAttribute("frequency", F32{100.0f}) == Result::SUCCESS);

            Modules::SignalAxes config;
            config.axes = "[C, S, B, _]";
            const auto module = CreateSignalAxesModule(implementation, config, input);
            const Tensor& output = module->outputs().at("buffer").tensor;

            REQUIRE(output.id() != input.id());
            REQUIRE(output.data() == input.data());
            REQUIRE(output.device() == input.device());
            REQUIRE(output.dtype() == input.dtype());
            REQUIRE(output.shape() == input.shape());
            REQUIRE(output.stride() == input.stride());
            REQUIRE(output.offset() == input.offset());
            REQUIRE(std::any_cast<F32>(output.attribute("frequency")) == 100.0f);

            REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == 0);
            REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == 1);
            REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == 2);

            REQUIRE(std::any_cast<Index>(input.attribute("sampleAxis")) == 3);
            REQUIRE(std::any_cast<Index>(input.attribute("batchAxis")) == 0);
            REQUIRE(std::any_cast<Index>(input.attribute("channelAxis")) == 2);
            REQUIRE(module->destroy() == Result::SUCCESS);

            Modules::SignalAxes leadingOnlyConfig;
            leadingOnlyConfig.axes = "[*]";
            const auto leadingOnly = CreateSignalAxesModule(
                implementation, leadingOnlyConfig, input);
            const Tensor& leadingOnlyOutput =
                leadingOnly->outputs().at("buffer").tensor;
            REQUIRE(std::any_cast<Index>(
                        leadingOnlyOutput.attribute("batchAxis")) == 0);
            REQUIRE_FALSE(leadingOnlyOutput.hasAttribute("channelAxis"));
            REQUIRE_FALSE(leadingOnlyOutput.hasAttribute("sampleAxis"));
            REQUIRE(leadingOnly->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal axes module inherits and clears roles",
          "[modules][signal_axes][metadata]") {
    const auto implementations = Registry::ListAvailableModules("signal_axes");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor implicitSamples(implementation.device, DataType::F32, {8});
            const auto inherited = CreateSignalAxesModule(
                implementation, Modules::SignalAxes{}, implicitSamples);
            const Tensor& inheritedOutput = inherited->outputs().at("buffer").tensor;
            REQUIRE_FALSE(inheritedOutput.hasAttribute("sampleAxis"));
            REQUIRE_FALSE(inheritedOutput.hasAttribute("batchAxis"));
            REQUIRE_FALSE(inheritedOutput.hasAttribute("channelAxis"));
            REQUIRE(inherited->destroy() == Result::SUCCESS);

            Modules::SignalAxes inheritImplicitConfig;
            inheritImplicitConfig.axes = "[*]";
            const auto inheritedImplicit = CreateSignalAxesModule(
                implementation, inheritImplicitConfig, implicitSamples);
            const Tensor& inheritedImplicitOutput =
                inheritedImplicit->outputs().at("buffer").tensor;
            REQUIRE_FALSE(inheritedImplicitOutput.hasAttribute("sampleAxis"));
            Jetstream::SignalAxes inheritedImplicitAxes;
            REQUIRE(ResolveSignalAxes(inheritedImplicitOutput,
                                      inheritedImplicitAxes) == Result::SUCCESS);
            REQUIRE(inheritedImplicitAxes.sample == Index{0});
            REQUIRE(inheritedImplicit->destroy() == Result::SUCCESS);

            Tensor explicitRoles(implementation.device, DataType::F32, {2, 8});
            REQUIRE(SetSignalAxes(explicitRoles, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);

            const auto inheritedExplicit = CreateSignalAxesModule(
                implementation, Modules::SignalAxes{}, explicitRoles);
            const Tensor& inheritedExplicitOutput =
                inheritedExplicit->outputs().at("buffer").tensor;
            REQUIRE(std::any_cast<Index>(
                        inheritedExplicitOutput.attribute("sampleAxis")) == 1);
            REQUIRE(std::any_cast<Index>(
                        inheritedExplicitOutput.attribute("batchAxis")) == 0);
            REQUIRE(inheritedExplicit->destroy() == Result::SUCCESS);

            Modules::SignalAxes partialConfig;
            partialConfig.axes = "[*, C]";
            const auto partial = CreateSignalAxesModule(
                implementation, partialConfig, explicitRoles);
            const Tensor& partialOutput = partial->outputs().at("buffer").tensor;
            REQUIRE_FALSE(partialOutput.hasAttribute("sampleAxis"));
            REQUIRE(std::any_cast<Index>(
                        partialOutput.attribute("batchAxis")) == 0);
            REQUIRE(std::any_cast<Index>(
                        partialOutput.attribute("channelAxis")) == 1);
            REQUIRE(partial->destroy() == Result::SUCCESS);

            Modules::SignalAxes inheritEachConfig;
            inheritEachConfig.axes = "[*, *]";
            const auto inheritedEach = CreateSignalAxesModule(
                implementation, inheritEachConfig, explicitRoles);
            const Tensor& inheritedEachOutput =
                inheritedEach->outputs().at("buffer").tensor;
            REQUIRE(std::any_cast<Index>(
                        inheritedEachOutput.attribute("sampleAxis")) == 1);
            REQUIRE(std::any_cast<Index>(
                        inheritedEachOutput.attribute("batchAxis")) == 0);
            REQUIRE(inheritedEach->destroy() == Result::SUCCESS);

            Modules::SignalAxes conflictConfig;
            conflictConfig.axes = "[S, *]";
            const auto conflict = CreateSignalAxesModule(
                implementation, conflictConfig, explicitRoles, Result::ERROR);
            REQUIRE(conflict->state() == Module::State::ERRORED);
            REQUIRE(conflict->outputs().empty());

            Modules::SignalAxes clearConfig;
            clearConfig.axes = "[_, _]";
            const auto cleared = CreateSignalAxesModule(
                implementation, clearConfig, explicitRoles);
            const Tensor& clearedOutput = cleared->outputs().at("buffer").tensor;
            REQUIRE_FALSE(clearedOutput.hasAttribute("sampleAxis"));
            REQUIRE_FALSE(clearedOutput.hasAttribute("batchAxis"));
            REQUIRE_FALSE(clearedOutput.hasAttribute("channelAxis"));
            REQUIRE(cleared->destroy() == Result::SUCCESS);

            Modules::SignalAxes clearRankOneConfig;
            clearRankOneConfig.axes = "[_]";
            const auto clearedRankOne = CreateSignalAxesModule(
                implementation, clearRankOneConfig, implicitSamples);
            const Tensor& clearedRankOneOutput =
                clearedRankOne->outputs().at("buffer").tensor;
            REQUIRE_FALSE(clearedRankOneOutput.hasAttribute("sampleAxis"));
            Jetstream::SignalAxes resolvedAxes;
            REQUIRE(ResolveSignalAxes(clearedRankOneOutput, resolvedAxes) ==
                    Result::SUCCESS);
            REQUIRE(resolvedAxes.sample == Index{0});
            REQUIRE(clearedRankOne->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal axes module creates pure channel layouts",
          "[modules][signal_axes][metadata][channel]") {
    const auto implementations = Registry::ListAvailableModules("signal_axes");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input(implementation.device, DataType::F32, {4, 16});
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);

            Modules::SignalAxes config;
            config.axes = "[B, C]";
            const auto module = CreateSignalAxesModule(implementation, config, input);
            const Tensor& output = module->outputs().at("buffer").tensor;
            REQUIRE_FALSE(output.hasAttribute("sampleAxis"));
            REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == 0);
            REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == 1);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal axes module validates layouts and can repair input metadata",
          "[modules][signal_axes][validation]") {
    const auto implementations = Registry::ListAvailableModules("signal_axes");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input(implementation.device, DataType::F32, {2, 8});
            for (const std::string& layout : {
                     "[]", "S", "[S, S]", "[X, S]", "[S,]", "[S,, _]",
                     "[B, C, S]",
                  }) {
                Modules::SignalAxes config;
                config.axes = layout;
                const auto module = CreateSignalAxesModule(
                    implementation, config, input, Result::ERROR);
                REQUIRE(module->state() == Module::State::ERRORED);
                REQUIRE(module->outputs().empty());
            }

            Tensor malformed(implementation.device, DataType::F32, {8});
            REQUIRE(malformed.setAttribute("sampleAxis", I64{0}) == Result::SUCCESS);
            const auto rejected = CreateSignalAxesModule(
                implementation, Modules::SignalAxes{}, malformed, Result::ERROR);
            REQUIRE(rejected->state() == Module::State::ERRORED);

            Modules::SignalAxes inheritMalformedConfig;
            inheritMalformedConfig.axes = "[*]";
            const auto inheritMalformed = CreateSignalAxesModule(
                implementation,
                inheritMalformedConfig,
                malformed,
                Result::ERROR);
            REQUIRE(inheritMalformed->state() == Module::State::ERRORED);

            Modules::SignalAxes repairConfig;
            repairConfig.axes = "[C]";
            const auto repaired = CreateSignalAxesModule(
                implementation, repairConfig, malformed);
            const Tensor& output = repaired->outputs().at("buffer").tensor;
            REQUIRE_FALSE(output.hasAttribute("sampleAxis"));
            REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == 0);
            REQUIRE(malformed.attribute("sampleAxis").type() == typeid(I64));
            REQUIRE(repaired->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal axes module accepts non-contiguous inputs and recreates on edits",
          "[modules][signal_axes][reconfigure]") {
    const auto implementations = Registry::ListAvailableModules("signal_axes");
    REQUIRE(!implementations.empty());

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            Tensor input(DeviceType::CPU, DataType::F32, {1, 8});
            REQUIRE(input.broadcastTo({4, 8}) == Result::SUCCESS);
            REQUIRE_FALSE(input.contiguous());
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .batch = Index{0},
            }) == Result::SUCCESS);

            TestContext context("signal_axes",
                                implementation.device,
                                implementation.runtime,
                                implementation.provider);
            Modules::SignalAxes config;
            config.axes = "\v";
            context.setConfig(config);
            context.setInput("buffer", input);
            REQUIRE(context.start() == Result::SUCCESS);
            REQUIRE(context.compute() == Result::SUCCESS);
            const Tensor& inheritedOutput = context.output("buffer");
            REQUIRE(std::any_cast<Index>(
                        inheritedOutput.attribute("sampleAxis")) == 1);
            REQUIRE(std::any_cast<Index>(
                        inheritedOutput.attribute("batchAxis")) == 0);

            config.axes = "[B, C]";
            REQUIRE(context.reconfigure(config) == Result::RECREATE);
            REQUIRE(context.stop() == Result::SUCCESS);
        }
    }
}
