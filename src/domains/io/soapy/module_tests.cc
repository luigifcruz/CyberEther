#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <SoapySDR/Errors.hpp>
#include <SoapySDR/Registry.hpp>

#include "jetstream/domains/io/soapy/module.hh"
#include "jetstream/registry.hh"
#include "module_impl.hh"

using namespace Jetstream;

namespace {

constexpr const char* TestSoapyDriver = "cyberether_test";

struct TestSoapyState {
    bool advertiseBiasTee = true;
    bool throwOnSettingInfo = false;
    bool failStreamSetup = false;
    std::vector<std::string> biasTeeWrites;
};

TestSoapyState testSoapyState;

class TestSoapyDevice final : public SoapySDR::Device {
 public:
    SoapySDR::RangeList getSampleRateRange(const int, const size_t) const override {
        return {SoapySDR::Range(1.0, 10.0e6)};
    }

    SoapySDR::RangeList getFrequencyRange(const int, const size_t) const override {
        return {SoapySDR::Range(1.0, 2.0e9)};
    }

    SoapySDR::ArgInfoList getSettingInfo() const override {
        if (testSoapyState.throwOnSettingInfo) {
            throw std::runtime_error("optional settings unavailable");
        }
        if (!testSoapyState.advertiseBiasTee) {
            return {};
        }

        SoapySDR::ArgInfo biasTee;
        biasTee.key = "biastee";
        biasTee.type = SoapySDR::ArgInfo::BOOL;
        return {biasTee};
    }

    void writeSetting(const std::string& key, const std::string& value) override {
        if (key == "biastee") {
            testSoapyState.biasTeeWrites.push_back(value);
        }
    }

    SoapySDR::Stream* setupStream(const int,
                                  const std::string&,
                                  const std::vector<size_t>&,
                                  const SoapySDR::Kwargs&) override {
        if (testSoapyState.failStreamSetup) {
            return nullptr;
        }
        return reinterpret_cast<SoapySDR::Stream*>(this);
    }

    int readStream(SoapySDR::Stream*,
                   void* const*,
                   const size_t,
                   int&,
                   long long&,
                   const long) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return SOAPY_SDR_TIMEOUT;
    }
};

SoapySDR::KwargsList FindTestSoapyDevice(const SoapySDR::Kwargs&) {
    return {{{"label", "CyberEther test device"}}};
}

SoapySDR::Device* MakeTestSoapyDevice(const SoapySDR::Kwargs&) {
    return new TestSoapyDevice();
}

const SoapySDR::Registry testSoapyRegistry(TestSoapyDriver,
                                           FindTestSoapyDevice,
                                           MakeTestSoapyDevice,
                                           SOAPY_SDR_ABI_VERSION);

struct SoapyImplAccess : Modules::SoapyImpl {
    static auto sampleRateRangesMember() {
        return &SoapyImplAccess::sampleRateRanges;
    }

    static auto frequencyRangesMember() {
        return &SoapyImplAccess::frequencyRanges;
    }
};

Modules::Soapy NonDefaultSoapyConfig() {
    Modules::Soapy config;
    config.modulePath = "/unused/soapy/module/path";
    config.deviceString = "driver=validation-must-precede-discovery";
    config.streamString = "bufflen=4096";
    config.frequency = 100.5e6f;
    config.sampleRate = 1.5e6f;
    config.automaticGain = false;
    config.biasTee = true;
    config.numberOfBatches = 3;
    config.numberOfTimeSamples = 17;
    config.bufferMultiplier = 2;
    return config;
}

Modules::Soapy TestDeviceSoapyConfig() {
    Modules::Soapy config;
    config.deviceString = std::string("driver=") + TestSoapyDriver;
    config.biasTee = true;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    config.bufferMultiplier = 1;
    return config;
}

std::shared_ptr<Module> BuildTestSoapyModule() {
    const auto implementations = Registry::ListAvailableModules("soapy");
    REQUIRE_FALSE(implementations.empty());

    const auto& implementation = implementations.front();
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("soapy",
                                  implementation.device,
                                  implementation.runtime,
                                  implementation.provider,
                                  module) == Result::SUCCESS);
    return module;
}

void RequireSoapyValidationError(const Registry::ModuleRegistration& impl,
                                 const Modules::Soapy& config) {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("soapy", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->outputs().empty());
    REQUIRE(module->outputs().empty());

    const auto defaults = Modules::Soapy{};
    const auto& applied = static_cast<const Modules::Soapy&>(module->config());
    REQUIRE(applied.modulePath == defaults.modulePath);
    REQUIRE(applied.deviceString == defaults.deviceString);
    REQUIRE(applied.streamString == defaults.streamString);
    REQUIRE(applied.frequency == defaults.frequency);
    REQUIRE(applied.sampleRate == defaults.sampleRate);
    REQUIRE(applied.automaticGain == defaults.automaticGain);
    REQUIRE(applied.biasTee == defaults.biasTee);
    REQUIRE(applied.numberOfBatches == defaults.numberOfBatches);
    REQUIRE(applied.numberOfTimeSamples == defaults.numberOfTimeSamples);
    REQUIRE(applied.bufferMultiplier == defaults.bufferMultiplier);
}

}  // namespace

TEST_CASE("Soapy module rejects candidates before hardware access and preserves staging",
          "[modules][soapy][validation][rollback]") {
    const auto implementations = Registry::ListAvailableModules("soapy");
    if (implementations.empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("frequency must be finite") {
                auto config = NonDefaultSoapyConfig();
                config.frequency = std::numeric_limits<F32>::quiet_NaN();
                RequireSoapyValidationError(impl, config);
            }

            SECTION("sample rate must be finite and positive") {
                auto config = NonDefaultSoapyConfig();
                config.sampleRate = std::numeric_limits<F32>::infinity();
                RequireSoapyValidationError(impl, config);

                config.sampleRate = 0.0f;
                RequireSoapyValidationError(impl, config);
            }

            SECTION("dimensions and multiplier must be nonzero") {
                auto config = NonDefaultSoapyConfig();
                config.numberOfBatches = 0;
                RequireSoapyValidationError(impl, config);

                config = NonDefaultSoapyConfig();
                config.numberOfTimeSamples = 0;
                RequireSoapyValidationError(impl, config);

                config = NonDefaultSoapyConfig();
                config.bufferMultiplier = 0;
                RequireSoapyValidationError(impl, config);
            }

            SECTION("output element product must not overflow") {
                auto config = NonDefaultSoapyConfig();
                config.numberOfBatches = std::numeric_limits<U64>::max();
                config.numberOfTimeSamples = 2;
                config.bufferMultiplier = 1;
                RequireSoapyValidationError(impl, config);
            }

            SECTION("output byte layout must not overflow") {
                auto config = NonDefaultSoapyConfig();
                config.numberOfBatches =
                    std::numeric_limits<U64>::max() / sizeof(CF32) + 1;
                config.numberOfTimeSamples = 1;
                config.bufferMultiplier = 1;
                RequireSoapyValidationError(impl, config);
            }

            SECTION("internal layout must not overflow") {
                auto config = NonDefaultSoapyConfig();
                config.numberOfBatches = 2;
                config.numberOfTimeSamples = 1;
                config.bufferMultiplier = std::numeric_limits<U64>::max();
                RequireSoapyValidationError(impl, config);

                config = NonDefaultSoapyConfig();
                config.numberOfBatches = 1;
                config.numberOfTimeSamples = 1;
                config.bufferMultiplier =
                    std::numeric_limits<U64>::max() / sizeof(CF32) + 1;
                RequireSoapyValidationError(impl, config);
            }

            SECTION("compile-target allocation must be representable") {
                auto config = NonDefaultSoapyConfig();
                config.numberOfBatches =
                    std::numeric_limits<U64>::max() / sizeof(CF32);
                config.numberOfTimeSamples = 1;
                config.bufferMultiplier = 1;
                RequireSoapyValidationError(impl, config);
            }
        }
    }
}

TEST_CASE("Soapy runtime ranges retain stepped capability checks",
          "[modules][soapy][devices]") {
    const std::vector ranges{SoapySDR::Range(1.0e6, 3.0e6, 1.0e6)};

    REQUIRE(Modules::SoapyRangeContains(ranges, 2.0e6f));
    REQUIRE_FALSE(Modules::SoapyRangeContains(ranges, 2.5e6f));

    const std::vector largeOffsetRange{
        SoapySDR::Range(0.0, 1.0e9, 1.0e9),
    };
    REQUIRE_FALSE(Modules::SoapyRangeContains(largeOffsetRange, 64.0f));

    const std::vector endpointRange{
        SoapySDR::Range(1000000000.1, 1000000000.1, 1.0),
    };
    const F32 endpoint = static_cast<F32>(endpointRange.front().minimum());
    REQUIRE(Modules::SoapyRangeContains(endpointRange, endpoint));
}

TEST_CASE("Soapy Bias-T follows the device lifecycle",
          "[modules][soapy][devices][bias-tee][lifecycle]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    testSoapyState = {};
    const auto config = TestDeviceSoapyConfig();

    SECTION("normal shutdown disables antenna power") {
        const auto module = BuildTestSoapyModule();
        REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        const auto writesBeforeDestroy = testSoapyState.biasTeeWrites;

        REQUIRE(module->destroy() == Result::SUCCESS);
        REQUIRE(writesBeforeDestroy == std::vector<std::string>{"true"});
        REQUIRE(testSoapyState.biasTeeWrites ==
                std::vector<std::string>{"true", "false"});
    }

    SECTION("creation failure disables antenna power") {
        testSoapyState.failStreamSetup = true;

        const auto module = BuildTestSoapyModule();
        REQUIRE(module->create("test", config, {}) == Result::ERROR);
        REQUIRE(testSoapyState.biasTeeWrites ==
                std::vector<std::string>{"true", "false"});
    }

    SECTION("optional capability failure does not reject the device") {
        testSoapyState.throwOnSettingInfo = true;

        const auto module = BuildTestSoapyModule();
        REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        REQUIRE(module->destroy() == Result::SUCCESS);
        REQUIRE(testSoapyState.biasTeeWrites.empty());
    }

    SECTION("unadvertised Bias-T is not written") {
        testSoapyState.advertiseBiasTee = false;

        const auto module = BuildTestSoapyModule();
        REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        REQUIRE(module->destroy() == Result::SUCCESS);
        REQUIRE(testSoapyState.biasTeeWrites.empty());
    }
}

TEST_CASE("Soapy validation ignores cached device ranges",
           "[modules][soapy][validation][devices]") {
    const auto implementations = Registry::ListAvailableModules("soapy");
    if (implementations.empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    for (const auto& implementation : implementations) {
        DYNAMIC_SECTION("Device: " << implementation.device
                        << " Runtime: " << implementation.runtime) {
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("soapy",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);

            auto* soapy = module->getImpl<Modules::SoapyImpl>();
            REQUIRE(soapy != nullptr);
            soapy->*SoapyImplAccess::sampleRateRangesMember() = {
                SoapySDR::Range(1.0e6, 2.0e6),
            };
            soapy->*SoapyImplAccess::frequencyRangesMember() = {
                SoapySDR::Range(90.0e6, 110.0e6),
            };

            auto& candidate = *soapy->candidate();
            candidate.sampleRate = 3.0e6f;
            candidate.frequency = 120.0e6f;

            REQUIRE_FALSE(Modules::SoapyRangeContains(
                soapy->*SoapyImplAccess::sampleRateRangesMember(), candidate.sampleRate));
            REQUIRE_FALSE(Modules::SoapyRangeContains(
                soapy->*SoapyImplAccess::frequencyRangesMember(), candidate.frequency));
            REQUIRE(soapy->validate() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Soapy device lists preserve duplicate and missing labels",
           "[modules][soapy][devices]") {
    const SoapySDR::KwargsList entries = {
        {{"driver", "rtlsdr"}, {"label", "RTL-SDR"}, {"serial", "A"}},
        {{"driver", "rtlsdr"}, {"label", "RTL-SDR"}, {"serial", "B"}},
        {{"driver", "rtlsdr"}, {"label", "RTL-SDR"}, {"serial", "B"}},
        {{"driver", "remote"}},
        {{"driver", "remote"}},
    };

    const auto devices = Modules::SoapyImpl::DeviceListFromEntries(entries);

    REQUIRE(devices.size() == entries.size());
    REQUIRE(devices.at("RTL-SDR").at("serial") == "A");
    REQUIRE(devices.at("RTL-SDR [B]").at("serial") == "B");
    REQUIRE(devices.at("RTL-SDR [B] #2").at("serial") == "B");
    REQUIRE(devices.contains("remote"));
    REQUIRE(devices.contains("remote #2"));
}
