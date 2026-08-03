#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>
#include <unordered_set>

#include "jetstream/domains/io/audio/module.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime.hh"
#include "jetstream/testing.hh"

#include "miniaudio.h"
#include "module_impl.hh"

using namespace Jetstream;

namespace {

struct AudioImplAccess : Modules::AudioImpl {
    static auto bufferMember() {
        return &AudioImplAccess::buffer;
    }

    static auto orderedInputMember() {
        return &AudioImplAccess::orderedInput;
    }

    static auto pendingInputMember() {
        return &AudioImplAccess::pendingInput;
    }
};

Tensor MakeAudioInput(const Registry::ModuleRegistration& impl,
                      const DataType dtype,
                      const U64 size,
                      const bool broadcast = false) {
    Tensor input;
    REQUIRE(input.create(impl.device, dtype, {broadcast ? 1 : size}) ==
            Result::SUCCESS);
    if (broadcast) {
        REQUIRE(input.broadcastTo({size}) == Result::SUCCESS);
    }
    return input;
}

void RequireAudioInputValidationError(
    const Registry::ModuleRegistration& impl,
    const Modules::Audio& config,
    const Tensor& input) {
    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("audio", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs().empty());
}

void RequireAudioValidationError(const Registry::ModuleRegistration& impl,
                                 const Modules::Audio& config,
                                 const DataType dtype = DataType::F32,
                                 const U64 inputSize = 64,
                                 const bool broadcast = false) {
    RequireAudioInputValidationError(
        impl, config, MakeAudioInput(impl, dtype, inputSize, broadcast));
}

TEST_CASE("Audio module rejects malformed or unsupported signal layouts",
          "[modules][audio][validation][layout]") {
    const auto implementations = Registry::ListAvailableModules("audio");
    if (implementations.empty()) {
        SUCCEED("Audio module is unavailable in this build.");
        return;
    }

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            Modules::Audio config;

            SECTION("multi-axis sample role is required") {
                Tensor input;
                REQUIRE(input.create(impl.device, DataType::F32, {2, 64}) ==
                        Result::SUCCESS);
                RequireAudioInputValidationError(impl, config, input);
            }

            SECTION("sample role type must be exact") {
                Tensor input;
                REQUIRE(input.create(impl.device, DataType::F32, {64}) ==
                        Result::SUCCESS);
                REQUIRE(input.setAttribute("sampleAxis", I64{0}) ==
                        Result::SUCCESS);
                RequireAudioInputValidationError(impl, config, input);
            }

            SECTION("roles must be distinct") {
                Tensor input = MakeAudioInput(impl, DataType::F32, 64);
                REQUIRE(input.setAttribute("batchAxis", Index{0}) ==
                        Result::SUCCESS);
                RequireAudioInputValidationError(impl, config, input);
            }

            SECTION("more than two audio channels are unsupported") {
                Tensor input;
                REQUIRE(input.create(impl.device, DataType::F32, {3, 64}) ==
                        Result::SUCCESS);
                REQUIRE(input.setAttribute("channelAxis", Index{0}) ==
                        Result::SUCCESS);
                REQUIRE(input.setAttribute("sampleAxis", Index{1}) ==
                        Result::SUCCESS);
                RequireAudioInputValidationError(impl, config, input);
            }

            SECTION("unidentified auxiliary dimensions are unsupported") {
                Tensor input;
                REQUIRE(input.create(impl.device, DataType::F32, {2, 64}) ==
                        Result::SUCCESS);
                REQUIRE(input.setAttribute("sampleAxis", Index{1}) ==
                        Result::SUCCESS);
                RequireAudioInputValidationError(impl, config, input);
            }
        }
    }
}

}  // namespace

TEST_CASE("Audio module interleaves batched planar stereo input",
          "[modules][audio][stereo]") {
    const auto implementations = Registry::ListAvailableModules("audio");
    if (implementations.empty()) {
        SUCCEED("Audio module is unavailable in this build.");
        return;
    }

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            Tensor input;
            REQUIRE(input.create(impl.device, DataType::F32, {2, 2, 4}) ==
                    Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{2},
                .batch = Index{0},
                .channel = Index{1},
            }) == Result::SUCCESS);
            const std::vector<F32> expected{
                1.0f, 11.0f, 2.0f, 12.0f, 3.0f, 13.0f, 4.0f, 14.0f,
                21.0f, 31.0f, 22.0f, 32.0f, 23.0f, 33.0f, 24.0f, 34.0f,
            };
            for (U64 batch = 0; batch < 2; ++batch) {
                for (U64 channel = 0; channel < 2; ++channel) {
                    for (U64 sample = 0; sample < 4; ++sample) {
                        input.data<F32>()[batch * 8 + channel * 4 + sample] =
                            static_cast<F32>(1 + batch * 20 +
                                             channel * 10 + sample);
                    }
                }
            }

            TensorMap inputs;
            inputs["buffer"].requested("test", "buffer");
            inputs["buffer"].tensor = input;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("audio", impl.device, impl.runtime,
                                          impl.provider, module) ==
                    Result::SUCCESS);
            Modules::Audio config;
            config.volume = 0.0f;
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            auto* audio = module->getImpl<Modules::AudioImpl>();
            REQUIRE(audio != nullptr);

            Runtime runtime("test", impl.device, impl.runtime);
            REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(runtime.compute({}, skippedModules, failedModules) ==
                    Result::SUCCESS);

            const auto& orderedInput =
                audio->*AudioImplAccess::orderedInputMember();
            const auto& output = audio->*AudioImplAccess::bufferMember();
            REQUIRE(orderedInput == expected);
            REQUIRE(output.size() == expected.size());

            ma_resampler_config referenceConfig = ma_resampler_config_init(
                ma_format_f32, 2, 48000, 48000,
                ma_resample_algorithm_linear);
            referenceConfig.linear.lpfOrder = 8;
            ma_resampler reference;
            REQUIRE(ma_resampler_init(&referenceConfig, nullptr, &reference) ==
                    MA_SUCCESS);
            ma_uint64 referenceFrameCountIn = expected.size() / 2;
            ma_uint64 referenceFrameCountOut = expected.size() / 2;
            std::vector<F32> referenceOutput(expected.size());
            REQUIRE(ma_resampler_process_pcm_frames(
                        &reference,
                        expected.data(),
                        &referenceFrameCountIn,
                        referenceOutput.data(),
                        &referenceFrameCountOut) == MA_SUCCESS);
            ma_resampler_uninit(&reference, nullptr);

            for (U64 index = 0; index < output.size(); ++index) {
                REQUIRE(output.at<F32>(index) == referenceOutput[index]);
            }
            REQUIRE((audio->*AudioImplAccess::pendingInputMember()).empty());

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Audio module preserves stereo frames across downsampling calls",
          "[modules][audio][stereo][resampling]") {
    const auto implementations = Registry::ListAvailableModules("audio");
    if (implementations.empty()) {
        SUCCEED("Audio module is unavailable in this build.");
        return;
    }

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            constexpr U64 framesPerChunk = 64;
            Tensor input;
            REQUIRE(input.create(impl.device, DataType::F32,
                                 {2, framesPerChunk}) == Result::SUCCESS);
            REQUIRE(SetSignalAxes(input, {
                .sample = Index{1},
                .channel = Index{0},
            }) == Result::SUCCESS);

            const auto fillChunk = [&](const U64 firstFrame) {
                for (U64 frame = 0; frame < framesPerChunk; ++frame) {
                    input.data<F32>()[frame] =
                        static_cast<F32>(firstFrame + frame);
                    input.data<F32>()[framesPerChunk + frame] =
                        static_cast<F32>(1000 + firstFrame + frame);
                }
            };
            fillChunk(0);

            TensorMap inputs;
            inputs["buffer"].requested("test", "buffer");
            inputs["buffer"].tensor = input;

            Modules::Audio config;
            config.inSampleRate = 48e3f;
            config.outSampleRate = 24e3f;
            config.volume = 0.0f;

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("audio", impl.device, impl.runtime,
                                          impl.provider, module) ==
                    Result::SUCCESS);
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            auto* audio = module->getImpl<Modules::AudioImpl>();
            REQUIRE(audio != nullptr);

            Runtime runtime("test", impl.device, impl.runtime);
            REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(runtime.compute({}, skippedModules, failedModules) ==
                    Result::SUCCESS);

            const auto& firstOutput = audio->*AudioImplAccess::bufferMember();
            std::vector<F32> chunkedOutput(firstOutput.data<F32>(),
                                           firstOutput.data<F32>() +
                                               firstOutput.size());
            const auto& firstPending =
                audio->*AudioImplAccess::pendingInputMember();
            REQUIRE(firstPending ==
                    std::vector<F32>{63.0f, 1063.0f});

            fillChunk(framesPerChunk);
            REQUIRE(runtime.compute({}, skippedModules, failedModules) ==
                    Result::SUCCESS);
            const auto& secondOutput = audio->*AudioImplAccess::bufferMember();
            chunkedOutput.insert(chunkedOutput.end(),
                                 secondOutput.data<F32>(),
                                 secondOutput.data<F32>() + secondOutput.size());

            std::vector<F32> continuousInput;
            continuousInput.reserve(framesPerChunk * 4);
            for (U64 frame = 0; frame < framesPerChunk * 2; ++frame) {
                continuousInput.push_back(static_cast<F32>(frame));
                continuousInput.push_back(static_cast<F32>(1000 + frame));
            }

            ma_resampler_config referenceConfig = ma_resampler_config_init(
                ma_format_f32, 2, 48000, 24000,
                ma_resample_algorithm_linear);
            referenceConfig.linear.lpfOrder = 8;
            ma_resampler reference;
            REQUIRE(ma_resampler_init(&referenceConfig, nullptr, &reference) ==
                    MA_SUCCESS);

            ma_uint64 referenceFrameCountIn = framesPerChunk * 2;
            ma_uint64 referenceFrameCountOut = framesPerChunk;
            std::vector<F32> referenceOutput(referenceFrameCountOut * 2);
            REQUIRE(ma_resampler_process_pcm_frames(
                        &reference,
                        continuousInput.data(),
                        &referenceFrameCountIn,
                        referenceOutput.data(),
                        &referenceFrameCountOut) == MA_SUCCESS);
            ma_resampler_uninit(&reference, nullptr);

            REQUIRE(referenceFrameCountIn == framesPerChunk * 2 - 1);
            REQUIRE(referenceFrameCountOut == framesPerChunk);
            REQUIRE(chunkedOutput.size() == referenceOutput.size());
            for (U64 index = 0; index < referenceOutput.size(); ++index) {
                REQUIRE_THAT(chunkedOutput[index],
                             Catch::Matchers::WithinAbs(
                                 referenceOutput[index], 1e-6f));
            }

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Audio module rejects invalid candidates before create",
          "[modules][audio][validation]") {
    auto implementations = Registry::ListAvailableModules("audio");
    if (implementations.empty()) {
        SUCCEED("Audio module is unavailable in this build.");
        return;
    }

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            SECTION("input rate must convert to a nonzero U32") {
                for (const F32 rate : {
                         -1.0f,
                         0.0f,
                         0.5f,
                         std::numeric_limits<F32>::infinity(),
                         std::numeric_limits<F32>::quiet_NaN(),
                         4294967296.0f,
                     }) {
                    Modules::Audio config;
                    config.inSampleRate = rate;
                    RequireAudioValidationError(impl, config);
                }
            }

            SECTION("output rate must convert to a nonzero U32") {
                for (const F32 rate : {
                         -1.0f,
                         0.0f,
                         0.5f,
                         std::numeric_limits<F32>::infinity(),
                         std::numeric_limits<F32>::quiet_NaN(),
                         4294967296.0f,
                     }) {
                    Modules::Audio config;
                    config.outSampleRate = rate;
                    RequireAudioValidationError(impl, config);
                }
            }

            SECTION("resampled output layout must not overflow") {
                Modules::Audio config;
                config.inSampleRate = 1.0f;
                config.outSampleRate = 4294967040.0f;
                const U64 rate = static_cast<U32>(config.outSampleRate);
                const U64 inputSize =
                    std::numeric_limits<U64>::max() / sizeof(F32) / rate + 1;
                RequireAudioValidationError(impl, config, DataType::F32,
                                            inputSize, true);
            }

            SECTION("circular buffer layout must not overflow") {
                Modules::Audio config;
                const U64 inputSize = std::numeric_limits<U64>::max() / 20 + 1;
                RequireAudioValidationError(impl, config, DataType::F32,
                                            inputSize, true);
            }

            SECTION("native CPU allocation layout must be representable") {
                Modules::Audio config;
                config.inSampleRate = 1.0f;
                config.outSampleRate = 20.0f;
                const U64 outputLimit =
                    std::numeric_limits<U64>::max() / sizeof(F32);
                RequireAudioValidationError(impl, config, DataType::F32,
                                            outputLimit / 20, true);
            }

            SECTION("native CPU input must be F32") {
                RequireAudioValidationError(impl, Modules::Audio{}, DataType::U8);
            }
        }
    }
}
