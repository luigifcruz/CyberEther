#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/psk_demod/module.hh"

#include "module_impl.hh"

#include <cmath>
#include <limits>
#include <unordered_set>

using namespace Jetstream;

namespace {

struct PskDemodImplAccess : Modules::PskDemodImpl {
    static auto frequencyErrorMember() {
        return &PskDemodImplAccess::frequencyError;
    }

    static auto frequencyBetaMember() {
        return &PskDemodImplAccess::freqBeta;
    }

    static auto frequencyAlphaMember() {
        return &PskDemodImplAccess::freqAlpha;
    }

    static auto timingAlphaMember() {
        return &PskDemodImplAccess::timingAlpha;
    }

    static auto timingBetaMember() {
        return &PskDemodImplAccess::timingBeta;
    }

    static auto laneStatesMember() {
        return &PskDemodImplAccess::laneStates;
    }
};

void RequirePskDemodValidationError(const Registry::ModuleRegistration& impl,
                                    const Modules::PskDemod& config,
                                    const TensorMap& inputs = {}) {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("psk_demod", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

TensorMap PskDemodInput(const Tensor& input) {
    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;
    return inputs;
}

}  // namespace

TEST_CASE("PskDemod - Costas frequency state uses integral gain",
          "[modules][psk_demod][costas]") {
    const auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CPU) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::PskDemod config;
            config.pskType = "bpsk";
            config.sampleRate = 2.0;
            config.symbolRate = 1.0;

            constexpr F32 phase = 0.2f;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {2}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) ==
                    Result::SUCCESS);
            input.at<CF32>(0) = std::polar(1.0f, phase);
            input.at<CF32>(1) = input.at<CF32>(0);

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("psk_demod", impl.device, impl.runtime,
                                          impl.provider, module) == Result::SUCCESS);
            REQUIRE(module->create("test", config, PskDemodInput(input)) ==
                    Result::SUCCESS);

            auto* psk = module->getImpl<Modules::PskDemodImpl>();
            REQUIRE(psk != nullptr);
            const F64 beta = psk->*PskDemodImplAccess::frequencyBetaMember();

            Runtime runtime("test", impl.device, impl.runtime);
            REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(runtime.compute({}, skippedModules, failedModules) ==
                    Result::SUCCESS);

            const F64 expected = beta * static_cast<F64>(input.at<CF32>(0).imag());
            const F64 actual = psk->*PskDemodImplAccess::frequencyErrorMember();
            REQUIRE_THAT(actual, Catch::Matchers::WithinAbs(expected, 1e-12));

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("PskDemod - Output Size Decimation", "[modules][psk_demod]") {
    auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

            Modules::PskDemod config;
            config.pskType = "qpsk";
            config.sampleRate = 2000000.0;
            config.symbolRate = 500000.0;

            ctx.setConfig(config);

            const U64 inputSize = 8192;
            const U64 expectedOutputSize = inputSize / 4;  // sampleRate / symbolRate = 4

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {inputSize}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) ==
                    Result::SUCCESS);

            for (U64 i = 0; i < inputSize; ++i) {
                input.at<CF32>(i) = CF32(1.0f, 0.0f);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            REQUIRE(out.size() == expectedOutputSize);
        }
    }
}

TEST_CASE("PskDemod - Output size uses the configured rate ratio",
          "[modules][psk_demod][timing][geometry]") {
    const auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

            Modules::PskDemod config;
            config.pskType = "bpsk";
            config.sampleRate = 2000000.0;
            config.symbolRate = 927000.0;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(impl.device, DataType::CF32, {4000}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleRate", F32{2000000.0f}) ==
                    Result::SUCCESS);
            for (U64 sample = 0; sample < input.size(); ++sample) {
                input.at<CF32>(sample) = CF32{1.0f, 0.0f};
            }
            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const Tensor& output = ctx.output("signal");
            REQUIRE(output.shape() == Shape{1854});
            REQUIRE(output.at<CF32>(output.size() - 1).real() > 0.0f);
            REQUIRE(output.attribute("sampleRate").type() == typeid(F32));
            REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) ==
                    927000.0f);

            TestContext exactCtx("psk_demod", impl.device, impl.runtime, impl.provider);
            config.sampleRate = 25.0;
            config.symbolRate = 7.0;
            exactCtx.setConfig(config);

            Tensor exactInput;
            REQUIRE(exactInput.create(impl.device, DataType::CF32, {25}) ==
                    Result::SUCCESS);
            REQUIRE(exactInput.setAttribute("sampleAxis", Index{0}) ==
                    Result::SUCCESS);
            for (U64 sample = 0; sample < exactInput.size(); ++sample) {
                exactInput.at<CF32>(sample) = CF32{1.0f, 0.0f};
            }
            exactCtx.setInput("signal", exactInput);

            REQUIRE(exactCtx.run() == Result::SUCCESS);
            REQUIRE(exactCtx.output("signal").shape() == Shape{7});
        }
    }
}

TEST_CASE("PskDemod - Fractional output waits for a complete tensor",
          "[modules][psk_demod][timing][pending][skip]") {
    const auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CPU) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::PskDemod config;
            config.pskType = "bpsk";
            config.sampleRate = 5.0;
            config.symbolRate = 2.0;

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {10}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleRate", F32{5.0f}) == Result::SUCCESS);
            const auto fillInput = [&](const F32 base) {
                for (U64 sample = 0; sample < input.size(); ++sample) {
                    input.at<CF32>(sample) =
                        CF32{base + static_cast<F32>(sample + 1), 0.0f};
                }
            };
            fillInput(0.0f);

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("psk_demod", impl.device, impl.runtime,
                                          impl.provider, module) == Result::SUCCESS);
            REQUIRE(module->create("test", config, PskDemodInput(input)) ==
                    Result::SUCCESS);

            auto* psk = module->getImpl<Modules::PskDemodImpl>();
            REQUIRE(psk != nullptr);
            auto& states = psk->*PskDemodImplAccess::laneStatesMember();
            REQUIRE(states.size() == 1);
            psk->*PskDemodImplAccess::frequencyAlphaMember() = 0.0;
            psk->*PskDemodImplAccess::frequencyBetaMember() = 0.0;
            psk->*PskDemodImplAccess::timingAlphaMember() = 0.0;
            psk->*PskDemodImplAccess::timingBetaMember() = 0.0;
            states.front().timingOmega = 3.0;

            Tensor output = module->outputs().at("signal").tensor;
            REQUIRE(output.shape() == Shape{4});
            REQUIRE(output.attribute("sampleRate").type() == typeid(F32));
            REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == 2.0f);
            const CF32 sentinel{-99.0f, -99.0f};
            for (U64 sample = 0; sample < output.size(); ++sample) {
                output.at<CF32>(sample) = sentinel;
            }

            Runtime runtime("test", impl.device, impl.runtime);
            REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(runtime.compute({}, skippedModules, failedModules) ==
                    Result::SUCCESS);
            REQUIRE(skippedModules.size() == 1);
            REQUIRE(skippedModules.contains("test"));
            REQUIRE(failedModules.empty());
            REQUIRE(states.front().pendingSymbols.size() == 3);
            for (U64 sample = 0; sample < output.size(); ++sample) {
                REQUIRE(output.at<CF32>(sample) == sentinel);
            }

            fillInput(100.0f);
            skippedModules.clear();
            failedModules.clear();
            REQUIRE(runtime.compute({}, skippedModules, failedModules) ==
                    Result::SUCCESS);
            REQUIRE(skippedModules.empty());
            REQUIRE(failedModules.empty());
            REQUIRE(output.at<CF32>(0) == CF32(1.0f, 0.0f));
            REQUIRE(output.at<CF32>(1) == CF32(4.0f, 0.0f));
            REQUIRE(output.at<CF32>(2) == CF32(7.0f, 0.0f));
            REQUIRE(output.at<CF32>(3) == CF32(10.0f, 0.0f));
            REQUIRE(states.front().pendingSymbols.size() == 3);

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("PskDemod - Incomplete lane skips the entire batched tensor",
          "[modules][psk_demod][timing][pending][skip][lanes]") {
    const auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CPU) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::PskDemod config;
            config.pskType = "bpsk";
            config.sampleRate = 4.0;
            config.symbolRate = 1.0;

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {10, 2, 2}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{2}) == Result::SUCCESS);
            const auto fillInput = [&](const F32 base) {
                for (U64 sample = 0; sample < input.shape(0); ++sample) {
                    for (U64 batch = 0; batch < input.shape(1); ++batch) {
                        input.at<CF32>(sample, batch, 0) = CF32{
                            base + static_cast<F32>(batch * 20 + sample + 1),
                            0.0f};
                        input.at<CF32>(sample, batch, 1) = CF32{
                            base + static_cast<F32>(batch * 20 + sample + 101),
                            0.0f};
                    }
                }
            };
            fillInput(0.0f);

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("psk_demod", impl.device, impl.runtime,
                                          impl.provider, module) == Result::SUCCESS);
            REQUIRE(module->create("test", config, PskDemodInput(input)) ==
                    Result::SUCCESS);

            auto* psk = module->getImpl<Modules::PskDemodImpl>();
            REQUIRE(psk != nullptr);
            auto& states = psk->*PskDemodImplAccess::laneStatesMember();
            REQUIRE(states.size() == 2);
            psk->*PskDemodImplAccess::frequencyAlphaMember() = 0.0;
            psk->*PskDemodImplAccess::frequencyBetaMember() = 0.0;
            psk->*PskDemodImplAccess::timingAlphaMember() = 0.0;
            psk->*PskDemodImplAccess::timingBetaMember() = 0.0;
            states[0].timingOmega = 2.0;
            states[1].timingOmega = 5.0;

            Tensor output = module->outputs().at("signal").tensor;
            REQUIRE(output.shape() == Shape{3, 2, 2});
            const CF32 sentinel{-99.0f, -99.0f};
            for (U64 sample = 0; sample < output.size(); ++sample) {
                output.data<CF32>()[sample] = sentinel;
            }

            Runtime runtime("test", impl.device, impl.runtime);
            REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(runtime.compute({}, skippedModules, failedModules) ==
                    Result::SUCCESS);
            REQUIRE(skippedModules.contains("test"));
            REQUIRE(states[0].pendingSymbols.size() == 10);
            REQUIRE(states[1].pendingSymbols.size() == 4);
            for (U64 sample = 0; sample < output.size(); ++sample) {
                REQUIRE(output.data<CF32>()[sample] == sentinel);
            }

            fillInput(1000.0f);
            skippedModules.clear();
            failedModules.clear();
            REQUIRE(runtime.compute({}, skippedModules, failedModules) ==
                    Result::SUCCESS);
            REQUIRE(skippedModules.empty());
            REQUIRE(output.at<CF32>(0, 0, 0) == CF32(1.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 0, 0) == CF32(3.0f, 0.0f));
            REQUIRE(output.at<CF32>(2, 0, 0) == CF32(5.0f, 0.0f));
            REQUIRE(output.at<CF32>(0, 1, 0) == CF32(7.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 1, 0) == CF32(9.0f, 0.0f));
            REQUIRE(output.at<CF32>(2, 1, 0) == CF32(21.0f, 0.0f));
            REQUIRE(output.at<CF32>(0, 0, 1) == CF32(101.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 0, 1) == CF32(106.0f, 0.0f));
            REQUIRE(output.at<CF32>(2, 0, 1) == CF32(121.0f, 0.0f));
            REQUIRE(output.at<CF32>(0, 1, 1) == CF32(126.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 1, 1) == CF32(1101.0f, 0.0f));
            REQUIRE(output.at<CF32>(2, 1, 1) == CF32(1106.0f, 0.0f));
            REQUIRE(states[0].pendingSymbols.size() == 14);
            REQUIRE(states[1].pendingSymbols.size() == 2);

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("PskDemod - Timing history stays bounded across submissions",
          "[modules][psk_demod][timing][history]") {
    const auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CPU) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::PskDemod config;
            config.pskType = "bpsk";
            config.sampleRate = 4.0;
            config.symbolRate = 1.0;

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {8, 3, 2}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
            for (U64 sample = 0; sample < 8; ++sample) {
                for (U64 batch = 0; batch < 3; ++batch) {
                    input.at<CF32>(sample, batch, 0) = CF32{1.0f, 0.0f};
                    input.at<CF32>(sample, batch, 1) = CF32{-1.0f, 0.0f};
                }
            }

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("psk_demod", impl.device, impl.runtime,
                                          impl.provider, module) == Result::SUCCESS);
            REQUIRE(module->create("test", config, PskDemodInput(input)) ==
                    Result::SUCCESS);

            auto* psk = module->getImpl<Modules::PskDemodImpl>();
            REQUIRE(psk != nullptr);
            auto& states = psk->*PskDemodImplAccess::laneStatesMember();
            REQUIRE(states.size() == 2);
            psk->*PskDemodImplAccess::frequencyAlphaMember() = 0.0;
            psk->*PskDemodImplAccess::frequencyBetaMember() = 0.0;
            psk->*PskDemodImplAccess::timingAlphaMember() = 0.0;
            psk->*PskDemodImplAccess::timingBetaMember() = 0.0;
            for (auto& state : states) {
                state.timingOmega = 2.0;
                state.frequencyError = 0.1;
                REQUIRE(state.sampleHistory.capacity() == 9);
                REQUIRE(state.pendingSymbols.capacity() == 54);
            }

            Runtime runtime("test", impl.device, impl.runtime);
            REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);
            for (U64 submission = 0; submission < 64; ++submission) {
                std::unordered_set<std::string> skippedModules;
                std::unordered_set<std::string> failedModules;
                REQUIRE(runtime.compute({}, skippedModules, failedModules) ==
                        Result::SUCCESS);

                for (U64 lane = 0; lane < states.size(); ++lane) {
                    CAPTURE(submission, lane);
                    REQUIRE(states[lane].sampleHistory.size() == 1);
                    REQUIRE(states[lane].sampleHistory.overflows() == 0);
                    REQUIRE(states[lane].pendingSymbols.size() == 6);
                    REQUIRE(states[lane].pendingSymbols.overflows() == 0);
                    CF32 retainedSample;
                    REQUIRE(states[lane].sampleHistory.peek(0, &retainedSample, 1) ==
                            Result::SUCCESS);
                    const CF32 expected = lane == 0
                        ? CF32{1.0f, 0.0f}
                        : CF32{-1.0f, 0.0f};
                    REQUIRE(retainedSample == expected);
                    REQUIRE_THAT(states[lane].timingMu,
                                 Catch::Matchers::WithinAbs(1.0, 1e-12));
                    REQUIRE(states[lane].hasLastSymbol);
                    if (submission == 0) {
                        REQUIRE_THAT(states[lane].phaseAccumulator,
                                     Catch::Matchers::WithinAbs(1.2, 1e-12));
                        states[lane].timingOmega = 4.0;
                    }
                }
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("PskDemod - Excess symbols are emitted in later submissions",
          "[modules][psk_demod][timing][pending]") {
    const auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CPU) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::PskDemod config;
            config.pskType = "bpsk";
            config.sampleRate = 4.0;
            config.symbolRate = 1.0;

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {8, 2, 2}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{2}) == Result::SUCCESS);
            const auto fillInput = [&](const F32 base) {
                for (U64 sample = 0; sample < 8; ++sample) {
                    for (U64 batch = 0; batch < 2; ++batch) {
                        for (U64 lane = 0; lane < 2; ++lane) {
                            input.at<CF32>(sample, batch, lane) = CF32{
                                base + static_cast<F32>(batch * 20 + lane * 10 + sample + 1),
                                0.0f};
                        }
                    }
                }
            };
            fillInput(0.0f);

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("psk_demod", impl.device, impl.runtime,
                                          impl.provider, module) == Result::SUCCESS);
            REQUIRE(module->create("test", config, PskDemodInput(input)) ==
                    Result::SUCCESS);

            auto* psk = module->getImpl<Modules::PskDemodImpl>();
            REQUIRE(psk != nullptr);
            auto& states = psk->*PskDemodImplAccess::laneStatesMember();
            REQUIRE(states.size() == 2);
            psk->*PskDemodImplAccess::frequencyAlphaMember() = 0.0;
            psk->*PskDemodImplAccess::frequencyBetaMember() = 0.0;
            psk->*PskDemodImplAccess::timingAlphaMember() = 0.0;
            psk->*PskDemodImplAccess::timingBetaMember() = 0.0;
            for (auto& state : states) {
                state.timingOmega = 2.0;
            }

            Runtime runtime("test", impl.device, impl.runtime);
            REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(runtime.compute({}, skippedModules, failedModules) ==
                    Result::SUCCESS);

            const Tensor output = module->outputs().at("signal").tensor;
            REQUIRE(output.shape() == Shape{2, 2, 2});
            REQUIRE(output.at<CF32>(0, 0, 0) == CF32(1.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 0, 0) == CF32(3.0f, 0.0f));
            REQUIRE(output.at<CF32>(0, 1, 0) == CF32(5.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 1, 0) == CF32(7.0f, 0.0f));
            REQUIRE(output.at<CF32>(0, 0, 1) == CF32(11.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 0, 1) == CF32(13.0f, 0.0f));
            REQUIRE(output.at<CF32>(0, 1, 1) == CF32(15.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 1, 1) == CF32(17.0f, 0.0f));
            for (const auto& state : states) {
                REQUIRE(state.pendingSymbols.size() == 4);
            }

            fillInput(100.0f);
            for (auto& state : states) {
                state.timingOmega = 4.0;
            }
            skippedModules.clear();
            failedModules.clear();
            REQUIRE(runtime.compute({}, skippedModules, failedModules) ==
                    Result::SUCCESS);
            REQUIRE(output.at<CF32>(0, 0, 0) == CF32(21.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 0, 0) == CF32(23.0f, 0.0f));
            REQUIRE(output.at<CF32>(0, 1, 0) == CF32(25.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 1, 0) == CF32(27.0f, 0.0f));
            REQUIRE(output.at<CF32>(0, 0, 1) == CF32(31.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 0, 1) == CF32(33.0f, 0.0f));
            REQUIRE(output.at<CF32>(0, 1, 1) == CF32(35.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 1, 1) == CF32(37.0f, 0.0f));
            for (const auto& state : states) {
                REQUIRE(state.pendingSymbols.size() == 4);
            }

            fillInput(200.0f);
            skippedModules.clear();
            failedModules.clear();
            REQUIRE(runtime.compute({}, skippedModules, failedModules) ==
                    Result::SUCCESS);
            REQUIRE(output.at<CF32>(0, 0, 0) == CF32(101.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 0, 0) == CF32(105.0f, 0.0f));
            REQUIRE(output.at<CF32>(0, 1, 0) == CF32(121.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 1, 0) == CF32(125.0f, 0.0f));
            REQUIRE(output.at<CF32>(0, 0, 1) == CF32(111.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 0, 1) == CF32(115.0f, 0.0f));
            REQUIRE(output.at<CF32>(0, 1, 1) == CF32(131.0f, 0.0f));
            REQUIRE(output.at<CF32>(1, 1, 1) == CF32(135.0f, 0.0f));
            for (auto& state : states) {
                REQUIRE(state.pendingSymbols.size() == 4);
                REQUIRE(state.pendingSymbols.overflows() == 0);
                state.timingOmega = 2.0;
            }

            Result overrunResult = Result::SUCCESS;
            for (U64 submission = 0;
                 submission < 16 && overrunResult == Result::SUCCESS;
                 ++submission) {
                skippedModules.clear();
                failedModules.clear();
                overrunResult = runtime.compute({}, skippedModules, failedModules);
            }
            REQUIRE(overrunResult == Result::ERROR);
            for (const auto& state : states) {
                REQUIRE(state.pendingSymbols.size() <=
                        state.pendingSymbols.capacity());
                REQUIRE(state.sampleHistory.size() <=
                        state.sampleHistory.capacity());
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("PskDemod - Independent Lanes And Ordered Batches",
          "[modules][psk_demod][layout][metadata]") {
    const auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

            Modules::PskDemod config;
            config.pskType = "bpsk";
            config.sampleRate = 2.0;
            config.symbolRate = 1.0;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(impl.device, DataType::CF32, {4, 2, 3, 2}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{2}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleRate", F32{2.0f}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("layoutMarker", U64{17}) == Result::SUCCESS);
            for (U64 sample = 0; sample < input.shape(0); ++sample) {
                for (U64 batch = 0; batch < input.shape(1); ++batch) {
                    for (U64 channel = 0; channel < input.shape(2); ++channel) {
                        for (U64 lane = 0; lane < input.shape(3); ++lane) {
                            input.at<CF32>(sample, batch, channel, lane) = CF32{
                                static_cast<F32>(1 + channel * 10 + lane),
                                0.0f};
                        }
                    }
                }
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const Tensor& out = ctx.output("signal");
            REQUIRE(out.shape() == Shape{2, 2, 3, 2});
            REQUIRE(std::any_cast<U64>(out.attribute("layoutMarker")) == U64{17});
            REQUIRE(out.attribute("sampleRate").type() == typeid(F32));
            REQUIRE(std::any_cast<F32>(out.attribute("sampleRate")) == 1.0f);
            for (U64 batch = 0; batch < 2; ++batch) {
                for (U64 channel = 0; channel < 3; ++channel) {
                    for (U64 lane = 0; lane < 2; ++lane) {
                        const CF32 expected{
                            static_cast<F32>(1 + channel * 10 + lane),
                            0.0f};
                        REQUIRE(out.at<CF32>(0, batch, channel, lane) ==
                                expected);
                        REQUIRE(out.at<CF32>(1, batch, channel, lane) ==
                                expected);
                    }
                }
            }

            SignalAxes axes;
            REQUIRE(ResolveSignalAxes(out, axes) == Result::SUCCESS);
            REQUIRE(axes.sample == Index{0});
            REQUIRE(axes.batch == Index{1});
            REQUIRE(axes.channel == Index{2});
        }
    }
}

TEST_CASE("PskDemod - BPSK Constant Phase", "[modules][psk_demod][bpsk]") {
    auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

            Modules::PskDemod config;
            config.pskType = "bpsk";
            config.sampleRate = 1000000.0;
            config.symbolRate = 250000.0;
            config.frequencyLoopBandwidth = 0.01;
            config.timingLoopBandwidth = 0.01;
            config.dampingFactor = 0.707;

            ctx.setConfig(config);

            const U64 inputSize = 1024;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {inputSize}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) ==
                    Result::SUCCESS);

            // Create constant positive phase BPSK signal.
            for (U64 i = 0; i < inputSize; ++i) {
                input.at<CF32>(i) = CF32(1.0f, 0.0f);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // After initial transient, output should be near positive real axis.
            for (U64 i = out.size() / 2; i < out.size(); ++i) {
                REQUIRE(out.at<CF32>(i).real() > 0.0f);
            }
        }
    }
}

TEST_CASE("PskDemod - QPSK Quadrants", "[modules][psk_demod][qpsk]") {
    auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

            Modules::PskDemod config;
            config.pskType = "qpsk";
            config.sampleRate = 2000000.0;
            config.symbolRate = 500000.0;
            config.frequencyLoopBandwidth = 0.05;
            config.timingLoopBandwidth = 0.05;
            config.dampingFactor = 0.707;

            ctx.setConfig(config);

            const U64 inputSize = 4096;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {inputSize}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) ==
                    Result::SUCCESS);

            // Create first quadrant QPSK signal.
            constexpr F32 INV_SQRT2 = 0.7071067811865475f;
            for (U64 i = 0; i < inputSize; ++i) {
                input.at<CF32>(i) = CF32(INV_SQRT2, INV_SQRT2);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // After transient, output should be in first quadrant.
            for (U64 i = out.size() / 2; i < out.size(); ++i) {
                REQUIRE(out.at<CF32>(i).real() > 0.0f);
                REQUIRE(out.at<CF32>(i).imag() > 0.0f);
            }
        }
    }
}

TEST_CASE("PskDemod - 8PSK Basic", "[modules][psk_demod][8psk]") {
    auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

            Modules::PskDemod config;
            config.pskType = "8psk";
            config.sampleRate = 4000000.0;
            config.symbolRate = 1000000.0;
            config.frequencyLoopBandwidth = 0.05;
            config.timingLoopBandwidth = 0.05;
            config.dampingFactor = 0.707;

            ctx.setConfig(config);

            const U64 inputSize = 8192;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {inputSize}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) ==
                    Result::SUCCESS);

            // Create 8-PSK signal at 0 degrees.
            for (U64 i = 0; i < inputSize; ++i) {
                input.at<CF32>(i) = CF32(1.0f, 0.0f);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // Check output is reasonable (non-zero magnitude).
            for (U64 i = out.size() / 2; i < out.size(); ++i) {
                const F32 mag = std::abs(out.at<CF32>(i));
                REQUIRE(mag > 0.1f);
            }
        }
    }
}

TEST_CASE("PskDemod - Invalid Configuration", "[modules][psk_demod][validation]") {
    auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("Symbol rate greater than sample rate") {
                TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

                Modules::PskDemod config;
                config.pskType = "qpsk";
                config.sampleRate = 1000000.0;
                config.symbolRate = 2000000.0;

                ctx.setConfig(config);

                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {1024}) == Result::SUCCESS);
                REQUIRE(input.setAttribute("sampleAxis", Index{0}) ==
                        Result::SUCCESS);

                ctx.setInput("signal", input);

                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("Negative sample rate") {
                TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

                Modules::PskDemod config;
                config.pskType = "qpsk";
                config.sampleRate = -1000000.0;
                config.symbolRate = 250000.0;

                ctx.setConfig(config);

                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {1024}) == Result::SUCCESS);
                REQUIRE(input.setAttribute("sampleAxis", Index{0}) ==
                        Result::SUCCESS);

                ctx.setInput("signal", input);

                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("Invalid loop bandwidth") {
                TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

                Modules::PskDemod config;
                config.pskType = "qpsk";
                config.sampleRate = 2000000.0;
                config.symbolRate = 500000.0;
                config.frequencyLoopBandwidth = 1.5;

                ctx.setConfig(config);

                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {1024}) == Result::SUCCESS);
                REQUIRE(input.setAttribute("sampleAxis", Index{0}) ==
                        Result::SUCCESS);

                ctx.setInput("signal", input);

                REQUIRE(ctx.run() == Result::ERROR);
            }
        }
    }
}

TEST_CASE("PskDemod - Direct configuration validation is input-phase independent",
          "[modules][psk_demod][validation][config]") {
    const auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("sample rate must be finite") {
                Modules::PskDemod config;
                config.sampleRate = std::numeric_limits<F64>::quiet_NaN();
                RequirePskDemodValidationError(impl, config);
            }

            SECTION("symbol rate must be finite") {
                Modules::PskDemod config;
                config.symbolRate = std::numeric_limits<F64>::infinity();
                RequirePskDemodValidationError(impl, config);
            }

            SECTION("symbol rate must fit output metadata") {
                constexpr F64 maxF32 =
                    static_cast<F64>(std::numeric_limits<F32>::max());
                Modules::PskDemod config;
                config.sampleRate = maxF32 * 4.0;
                config.symbolRate = maxF32 * 2.0;
                RequirePskDemodValidationError(impl, config);
            }

            SECTION("frequency bandwidth must be finite") {
                Modules::PskDemod config;
                config.frequencyLoopBandwidth =
                    std::numeric_limits<F64>::quiet_NaN();
                RequirePskDemodValidationError(impl, config);
            }

            SECTION("timing bandwidth must be finite") {
                Modules::PskDemod config;
                config.timingLoopBandwidth = std::numeric_limits<F64>::infinity();
                RequirePskDemodValidationError(impl, config);
            }

            SECTION("damping factor must be finite") {
                Modules::PskDemod config;
                config.dampingFactor = std::numeric_limits<F64>::quiet_NaN();
                RequirePskDemodValidationError(impl, config);
            }

            SECTION("effective samples per symbol must be at least two") {
                Modules::PskDemod config;
                config.sampleRate = 1999999.0;
                config.symbolRate = 1000000.0;
                RequirePskDemodValidationError(impl, config);
            }

            SECTION("samples per symbol must be safely representable") {
                Modules::PskDemod config;
                config.sampleRate = std::numeric_limits<F64>::max();
                config.symbolRate = std::numeric_limits<F64>::min();
                RequirePskDemodValidationError(impl, config);
            }

            SECTION("loop coefficients must remain usable") {
                Modules::PskDemod config;
                config.frequencyLoopBandwidth =
                    std::numeric_limits<F64>::denorm_min();
                RequirePskDemodValidationError(impl, config);
            }

            SECTION("constellation must be supported") {
                Modules::PskDemod config;
                config.pskType = "16psk";
                RequirePskDemodValidationError(impl, config);
            }
        }
    }
}

TEST_CASE("PskDemod - Direct input metadata validation precedes create",
          "[modules][psk_demod][validation][input]") {
    const auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Modules::PskDemod config;

            SECTION("rank zero") {
                Tensor input;
                REQUIRE(input.create(impl.device, DataType::CF32, {1}) ==
                        Result::SUCCESS);
                REQUIRE(input.squeezeDims(0) == Result::SUCCESS);
                REQUIRE(input.rank() == 0);
                RequirePskDemodValidationError(impl, config, PskDemodInput(input));
            }

            SECTION("missing sample role on rank two") {
                Tensor input;
                REQUIRE(input.create(impl.device, DataType::CF32, {2, 2}) ==
                        Result::SUCCESS);
                RequirePskDemodValidationError(impl, config, PskDemodInput(input));
            }

            SECTION("sample role type must be exact") {
                Tensor input;
                REQUIRE(input.create(impl.device, DataType::CF32, {8}) ==
                        Result::SUCCESS);
                REQUIRE(input.setAttribute("sampleAxis", I64{0}) ==
                        Result::SUCCESS);
                RequirePskDemodValidationError(impl, config, PskDemodInput(input));
            }

            SECTION("sample role must be in range") {
                Tensor input;
                REQUIRE(input.create(impl.device, DataType::CF32, {8}) ==
                        Result::SUCCESS);
                REQUIRE(input.setAttribute("sampleAxis", Index{1}) ==
                        Result::SUCCESS);
                RequirePskDemodValidationError(impl, config, PskDemodInput(input));
            }

            SECTION("roles must be distinct") {
                Tensor input;
                REQUIRE(input.create(impl.device, DataType::CF32, {8}) ==
                        Result::SUCCESS);
                REQUIRE(input.setAttribute("sampleAxis", Index{0}) ==
                        Result::SUCCESS);
                REQUIRE(input.setAttribute("batchAxis", Index{0}) ==
                        Result::SUCCESS);
                RequirePskDemodValidationError(impl, config, PskDemodInput(input));
            }

            SECTION("insufficient backing capacity") {
                Tensor input;
                REQUIRE(input.create(impl.device, DataType::CF32, {1}) ==
                        Result::SUCCESS);
                REQUIRE(input.broadcastTo({8}) == Result::SUCCESS);
                REQUIRE(input.rank() == 1);
                REQUIRE(input.sizeBytes() > input.buffer().sizeBytes());
                REQUIRE(input.setAttribute("sampleAxis", Index{0}) ==
                        Result::SUCCESS);
                RequirePskDemodValidationError(impl, config, PskDemodInput(input));
            }

            SECTION("unsupported dtype") {
                Tensor input;
                REQUIRE(input.create(impl.device, DataType::F64, {8}) ==
                        Result::SUCCESS);
                REQUIRE(input.setAttribute("sampleAxis", Index{0}) ==
                        Result::SUCCESS);
                RequirePskDemodValidationError(impl, config, PskDemodInput(input));
            }

            if constexpr (sizeof(std::size_t) < sizeof(U64)) {
                SECTION("CPU allocation size") {
                    const U64 inputSize =
                        (static_cast<U64>(std::numeric_limits<std::size_t>::max()) /
                         sizeof(CF32)) * 4;
                    Tensor input;
                    REQUIRE(input.create(reinterpret_cast<void*>(0x1000),
                                         impl.device,
                                         DataType::CF32,
                                         {inputSize}) == Result::SUCCESS);
                    REQUIRE(input.setAttribute("sampleAxis", Index{0}) ==
                            Result::SUCCESS);
                    RequirePskDemodValidationError(impl, config, PskDemodInput(input));
                }
            }
        }
    }
}

TEST_CASE("PskDemod - Exact ratio two is valid",
          "[modules][psk_demod][validation][boundary]") {
    const auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

            Modules::PskDemod config;
            config.sampleRate = 2000000.0;
            config.symbolRate = 1000000.0;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {8}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) ==
                    Result::SUCCESS);
            for (U64 i = 0; i < input.size(); ++i) {
                input.at<CF32>(i) = CF32(1.0f, 0.0f);
            }
            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE(ctx.output("signal").shape() == Shape{4});
        }
    }
}
