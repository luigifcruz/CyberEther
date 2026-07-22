#include <catch2/catch_test_macros.hpp>

#include <limits>

#include "jetstream/domains/io/soapy/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"
#include "module_impl.hh"

using namespace Jetstream;

TEST_CASE("Soapy module rejects invalid batch dimensions",
          "[modules][soapy][validation]") {
    auto implementations = Registry::ListAvailableModules("soapy");
    if (implementations.empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    for (const auto& impl : implementations) {
        SECTION("numberOfBatches must be > 0") {
            TestContext ctx("soapy", impl.device, impl.runtime, impl.provider);

            Modules::Soapy config;
            config.numberOfBatches = 0;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("numberOfTimeSamples must be > 0") {
            TestContext ctx("soapy", impl.device, impl.runtime, impl.provider);

            Modules::Soapy config;
            config.numberOfTimeSamples = 0;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("bufferMultiplier must be > 0") {
            TestContext ctx("soapy", impl.device, impl.runtime, impl.provider);

            Modules::Soapy config;
            config.bufferMultiplier = 0;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("non-default params still validate dimensions") {
            TestContext ctx("soapy", impl.device, impl.runtime, impl.provider);

            Modules::Soapy config;
            config.deviceString = "driver=mock";
            config.streamString = "bufflen=4096";
            config.frequency = 100.5e6f;
            config.sampleRate = 1.5e6f;
            config.automaticGain = false;
            config.numberOfBatches = 0;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}

TEST_CASE("Soapy module validates scalar configuration before device access",
          "[modules][soapy][validation]") {
    Modules::Soapy config;
    const auto validate = [&] {
        return Modules::ValidateSoapyConfig(config.frequency,
                                            config.sampleRate,
                                            config.numberOfBatches,
                                            config.numberOfTimeSamples,
                                            config.bufferMultiplier);
    };

    SECTION("frequency must be finite") {
        config.frequency = std::numeric_limits<F32>::quiet_NaN();
        REQUIRE(validate() == Result::ERROR);
    }

    SECTION("sample rate must be finite and positive") {
        config.sampleRate = -1.0f;
        REQUIRE(validate() == Result::ERROR);

        config.sampleRate = std::numeric_limits<F32>::infinity();
        REQUIRE(validate() == Result::ERROR);
    }

    SECTION("buffer dimensions must not overflow") {
        config.numberOfBatches = std::numeric_limits<U64>::max();
        REQUIRE(validate() == Result::ERROR);
    }

    SECTION("internal buffer dimensions must not overflow") {
        config.bufferMultiplier = std::numeric_limits<U64>::max();
        REQUIRE(validate() == Result::ERROR);
    }
}

TEST_CASE("Soapy module validates same-device candidates against cached capabilities",
          "[modules][soapy][validation][reconfigure]") {
    Modules::Soapy config;
    config.sampleRate = 1.5e6f;
    config.frequency = 100.0e6f;
    const std::vector sampleRates{SoapySDR::Range(1.0e6, 2.0e6)};
    const std::vector frequencies{SoapySDR::Range(90.0e6, 110.0e6)};
    const auto validate = [&](const bool useCachedRanges) {
        return Modules::ValidateSoapyConfig(config.frequency,
                                            config.sampleRate,
                                            config.numberOfBatches,
                                            config.numberOfTimeSamples,
                                            config.bufferMultiplier,
                                            useCachedRanges ? &sampleRates : nullptr,
                                            useCachedRanges ? &frequencies : nullptr);
    };
    REQUIRE(validate(true) == Result::SUCCESS);

    SECTION("unsupported sample rate is rejected") {
        config.sampleRate = 3.0e6f;
        REQUIRE(validate(true) == Result::ERROR);
    }

    SECTION("unsupported frequency is rejected") {
        config.frequency = 120.0e6f;
        REQUIRE(validate(true) == Result::ERROR);
    }

    SECTION("sample rate must match a stepped range") {
        const std::vector steppedSampleRates{SoapySDR::Range(1.0e6, 3.0e6, 1.0e6)};
        config.sampleRate = 2.0e6f;
        REQUIRE(Modules::ValidateSoapyConfig(config.frequency,
                                             config.sampleRate,
                                             config.numberOfBatches,
                                             config.numberOfTimeSamples,
                                             config.bufferMultiplier,
                                             &steppedSampleRates,
                                             &frequencies) == Result::SUCCESS);

        config.sampleRate = 2.5e6f;
        REQUIRE(Modules::ValidateSoapyConfig(config.frequency,
                                             config.sampleRate,
                                             config.numberOfBatches,
                                             config.numberOfTimeSamples,
                                             config.bufferMultiplier,
                                             &steppedSampleRates,
                                             &frequencies) == Result::ERROR);
    }

    SECTION("stepped ranges do not accept magnitude-scaled offsets") {
        const std::vector steppedSampleRates{SoapySDR::Range(0.0, 1.0e9, 1.0e9)};
        config.sampleRate = 64.0f;
        REQUIRE(Modules::ValidateSoapyConfig(config.frequency,
                                             config.sampleRate,
                                             config.numberOfBatches,
                                             config.numberOfTimeSamples,
                                             config.bufferMultiplier,
                                             &steppedSampleRates,
                                             &frequencies) == Result::ERROR);
    }

    SECTION("range endpoints are compared at configuration precision") {
        const std::vector steppedSampleRates{
            SoapySDR::Range(1000000000.1, 1000000000.1, 1.0),
        };
        config.sampleRate = static_cast<F32>(steppedSampleRates.front().minimum());
        REQUIRE(Modules::ValidateSoapyConfig(config.frequency,
                                             config.sampleRate,
                                             config.numberOfBatches,
                                             config.numberOfTimeSamples,
                                             config.bufferMultiplier,
                                             &steppedSampleRates,
                                             &frequencies) == Result::SUCCESS);
    }

    SECTION("capabilities from another device are ignored") {
        config.sampleRate = 3.0e6f;
        config.frequency = 120.0e6f;
        REQUIRE(validate(false) == Result::SUCCESS);
    }
}
