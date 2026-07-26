#include <catch2/catch_test_macros.hpp>

#include <limits>

#include "jetstream/domains/io/audio/module.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

namespace {

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

void RequireAudioValidationError(const Registry::ModuleRegistration& impl,
                                 const Modules::Audio& config,
                                 const DataType dtype = DataType::F32,
                                 const U64 inputSize = 64,
                                 const bool broadcast = false) {
    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = MakeAudioInput(impl, dtype, inputSize, broadcast);

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("audio", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->inputs().empty());
}

}  // namespace

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
