#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <any>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <SoapySDR/Errors.hpp>
#include <SoapySDR/Registry.hpp>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/io/soapy/block.hh"
#include "jetstream/domains/io/soapy/module.hh"
#include "jetstream/module_context.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "jetstream/scheduler_context.hh"
#include "module_impl.hh"
#include "receive_status.hh"

using namespace Jetstream;

namespace {

constexpr const char* TestSoapyDriver = "cyberether_test";

struct TestSoapyReads {
    std::vector<int> results;
    std::atomic<size_t> calls{0};
    bool throwOnRead = false;
};

struct TestSoapyState {
    bool advertiseBiasTee = true;
    bool throwOnSettingInfo = false;
    bool failStreamSetup = false;
    std::vector<std::string> biasTeeWrites;
    std::shared_ptr<TestSoapyReads> reads = std::make_shared<TestSoapyReads>();
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
                   void* const* buffers,
                   const size_t numElems,
                   int&,
                   long long&,
                   const long) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const auto index = reads->calls.fetch_add(1);
        if (reads->throwOnRead) {
            throw std::runtime_error("test receive failure");
        }
        const int result = index < reads->results.size()
            ? reads->results[index] : SOAPY_SDR_TIMEOUT;
        if (result > 0) {
            std::fill_n(static_cast<CF32*>(buffers[0]),
                        std::min(numElems, static_cast<size_t>(result)),
                        CF32{static_cast<F32>(index), -1.0f});
        }
        return result;
    }

 private:
    const std::shared_ptr<TestSoapyReads> reads = testSoapyState.reads;
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

    static auto erroredMember() {
        return &SoapyImplAccess::errored;
    }

    static auto circularBufferMember() {
        return &SoapyImplAccess::circularBuffer;
    }
};

struct SoapyLogCapture {
    std::ostringstream sink;
    std::ostream* previousSink = &JST_LOG_SINK();

    SoapyLogCapture() {
        JST_LOG_SET_SINK(&sink);
    }

    ~SoapyLogCapture() {
        JST_LOG_SET_SINK(previousSink);
    }

    std::string text() const {
        std::lock_guard lock(_JST_LOG_MUTEX());
        return sink.str();
    }
};

struct SoapyModuleCleanup {
    std::shared_ptr<Module> module;

    ~SoapyModuleCleanup() {
        if (module->state() == Module::State::CREATED) {
            static_cast<void>(module->destroy());
        }
    }
};

template<typename Predicate>
bool WaitForSoapy(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

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

TEST_CASE("Soapy receive statuses distinguish idle reads from failures",
          "[modules][soapy][receive]") {
    using Status = detail::SoapyReceiveStatus;
    using Action = Status::Action;
    const auto start = Status::Clock::time_point{};
    Status status;

    REQUIRE(status.handle(8, start) == Action::Samples);
    REQUIRE(status.handle(0, start) == Action::Retry);
    REQUIRE(status.handle(SOAPY_SDR_TIMEOUT, start) == Action::Retry);
    REQUIRE(status.deviceOverflows == 0);

    for (const int error : {SOAPY_SDR_STREAM_ERROR, SOAPY_SDR_CORRUPTION,
                           SOAPY_SDR_NOT_SUPPORTED, SOAPY_SDR_TIME_ERROR,
                           SOAPY_SDR_UNDERFLOW, -999}) {
        REQUIRE(status.handle(error, start) == Action::Fail);
    }
    REQUIRE(status.deviceOverflows == 0);
}

TEST_CASE("Soapy device overflow warnings are counted and rate limited per stream",
          "[modules][soapy][receive]") {
    using Status = detail::SoapyReceiveStatus;
    using Action = Status::Action;
    using namespace std::chrono_literals;
    const auto start = Status::Clock::time_point{};
    Status status;

    REQUIRE(status.handle(SOAPY_SDR_OVERFLOW, start) == Action::WarnOverflow);
    REQUIRE(status.deviceOverflows == 1);
    for (int i = 0; i < 100; ++i) {
        REQUIRE(status.handle(SOAPY_SDR_OVERFLOW, start + 999ms) == Action::Retry);
    }
    REQUIRE(status.deviceOverflows == 101);

    REQUIRE(status.handle(8, start + 999ms) == Action::Samples);
    REQUIRE(status.handle(SOAPY_SDR_TIMEOUT, start + 999ms) == Action::Retry);
    REQUIRE(status.handle(SOAPY_SDR_OVERFLOW, start + 1s) == Action::WarnOverflow);
    REQUIRE(status.deviceOverflows == 102);
    REQUIRE(status.handle(SOAPY_SDR_OVERFLOW, start + 1999ms) == Action::Retry);
    REQUIRE(status.handle(SOAPY_SDR_OVERFLOW, start + 2s) == Action::WarnOverflow);
    REQUIRE(status.deviceOverflows == 104);

    Status otherStream;
    REQUIRE(otherStream.handle(SOAPY_SDR_OVERFLOW, start) == Action::WarnOverflow);
    REQUIRE(otherStream.deviceOverflows == 1);
}

TEST_CASE("Soapy receiver resumes samples after timeouts and device overflows",
          "[modules][soapy][receive]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    testSoapyState = {};
    const auto reads = testSoapyState.reads;
    reads->results = {SOAPY_SDR_TIMEOUT, 0, SOAPY_SDR_OVERFLOW, SOAPY_SDR_OVERFLOW, 8};
    SoapyLogCapture logs;
    const auto module = BuildTestSoapyModule();
    const SoapyModuleCleanup cleanup{module};
    REQUIRE(module->create("test", TestDeviceSoapyConfig(), {}) == Result::SUCCESS);
    REQUIRE(WaitForSoapy([&] { return reads->calls > reads->results.size(); }));

    auto* soapy = module->getImpl<Modules::SoapyImpl>();
    REQUIRE_FALSE((soapy->*SoapyImplAccess::erroredMember()).load());
    REQUIRE((soapy->*SoapyImplAccess::circularBufferMember()).overflows() == 0);
    REQUIRE(module->context()->scheduler()->hasPendingCompute() == Result::SUCCESS);
    const auto runtime = std::dynamic_pointer_cast<NativeCpuRuntimeContext>(
        module->context()->runtime());
    REQUIRE(runtime != nullptr);
    REQUIRE(runtime->computeSubmit() == Result::SUCCESS);
    const auto& output = module->outputs().at("signal").tensor;
    const auto* data = static_cast<const CF32*>(output.data());
    for (U64 i = 0; i < output.size(); ++i) {
        REQUIRE(data[i] == CF32{4.0f, -1.0f});
    }
    REQUIRE(module->context()->scheduler()->hasPendingCompute() == Result::TIMEOUT);
    REQUIRE_FALSE((soapy->*SoapyImplAccess::erroredMember()).load());
    REQUIRE(module->destroy() == Result::SUCCESS);

    const auto text = logs.text();
    const auto warning = text.find("Device receive overflow");
    REQUIRE(warning != std::string::npos);
    REQUIRE(text.find("Device receive overflow", warning + 1) == std::string::npos);
    REQUIRE(text.find("total events since stream start: 1") != std::string::npos);
    REQUIRE(text.find("TIMEOUT") == std::string::npos);
    REQUIRE(text.find("Failed to read stream") == std::string::npos);
}

TEST_CASE("Soapy receiver stops on read failures and reports them to the scheduler",
          "[modules][soapy][receive]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    for (const int error : {SOAPY_SDR_STREAM_ERROR, SOAPY_SDR_CORRUPTION,
                           SOAPY_SDR_NOT_SUPPORTED, SOAPY_SDR_TIME_ERROR,
                           SOAPY_SDR_UNDERFLOW, -999, 9}) {
        DYNAMIC_SECTION("Read result: " << error) {
            testSoapyState = {};
            const auto reads = testSoapyState.reads;
            reads->results = {error, 8};
            SoapyLogCapture logs;
            const auto module = BuildTestSoapyModule();
            const SoapyModuleCleanup cleanup{module};
            REQUIRE(module->create("test", TestDeviceSoapyConfig(), {}) == Result::SUCCESS);
            auto* soapy = module->getImpl<Modules::SoapyImpl>();
            REQUIRE(WaitForSoapy([&] {
                return (soapy->*SoapyImplAccess::erroredMember()).load();
            }));

            REQUIRE(module->context()->scheduler()->hasPendingCompute() == Result::ERROR);
            const auto runtime = std::dynamic_pointer_cast<NativeCpuRuntimeContext>(
                module->context()->runtime());
            REQUIRE(runtime != nullptr);
            REQUIRE(runtime->computeSubmit() == Result::ERROR);
            REQUIRE(module->destroy() == Result::SUCCESS);
            REQUIRE(reads->calls == 1);
            if (error < 0) {
                REQUIRE(logs.text().find(SoapySDR::errToStr(error)) != std::string::npos);
                REQUIRE(logs.text().find("(" + std::to_string(error) + ")") != std::string::npos);
                REQUIRE(logs.text().find("Stopping reception") != std::string::npos);
            } else {
                REQUIRE(logs.text().find("more samples than requested") != std::string::npos);
            }
        }
    }
}

TEST_CASE("Soapy receiver contains driver exceptions",
          "[modules][soapy][receive]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    testSoapyState = {};
    testSoapyState.reads->throwOnRead = true;
    SoapyLogCapture logs;
    const auto module = BuildTestSoapyModule();
    const SoapyModuleCleanup cleanup{module};
    REQUIRE(module->create("test", TestDeviceSoapyConfig(), {}) == Result::SUCCESS);
    auto* soapy = module->getImpl<Modules::SoapyImpl>();
    REQUIRE(WaitForSoapy([&] {
        return (soapy->*SoapyImplAccess::erroredMember()).load();
    }));
    REQUIRE(module->context()->scheduler()->hasPendingCompute() == Result::ERROR);
    REQUIRE(module->destroy() == Result::SUCCESS);
    REQUIRE(testSoapyState.reads->calls == 1);
    REQUIRE(logs.text().find("test receive failure") != std::string::npos);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Soapy buffer loss metric counts discarded samples and survives draining",
                 "[modules][soapy][receive][metrics]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    testSoapyState = {};
    const auto reads = testSoapyState.reads;
    Blocks::Soapy config;
    config.hintString = std::string("driver=") + TestSoapyDriver;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    config.bufferMultiplier = 1;
    std::pair<std::string, F32> expected;

    SECTION("partial and full overwrites count samples rather than events") {
        reads->results = {SOAPY_SDR_TIMEOUT, SOAPY_SDR_OVERFLOW, 8, 8, 4};
        expected = {"60.00%", 0.6f};
    }

    SECTION("loss percentage uses two decimal places without a sample count") {
        config.numberOfTimeSamples = 7560;
        config.bufferMultiplier = 3;
        reads->results = {8192, 8192, 7320};
        expected = {"4.32%", static_cast<F32>(1024.0 / 23704.0)};
    }

    SECTION("device overflows and timeouts do not increase application loss") {
        reads->results = {SOAPY_SDR_TIMEOUT, SOAPY_SDR_OVERFLOW, 8};
        expected = {"0.00%", 0.0f};
    }

    SECTION("small losses remain visible in the numeric label") {
        config.numberOfTimeSamples = 8192;
        config.bufferMultiplier = 2;
        reads->results = {8192, 8192, 1};
        expected = {"<0.01%", 1.0f / 16385.0f};
    }

    REQUIRE(flowgraph->blockCreate("radio", config, {}) == Result::SUCCESS);
    REQUIRE(WaitForSoapy([&] { return reads->calls > reads->results.size(); }));

    const auto metric = [&] {
        std::vector<Flowgraph::View::MetricEntry> metrics;
        REQUIRE(flowgraph->view().metrics("radio", metrics) == Result::SUCCESS);
        REQUIRE(std::none_of(metrics.begin(), metrics.end(), [](const auto& entry) {
            return entry.name == "bufferOverruns";
        }));
        const auto it = std::find_if(metrics.begin(), metrics.end(), [](const auto& entry) {
            return entry.name == "bufferLoss";
        });
        REQUIRE(it != metrics.end());
        REQUIRE(it->label == "Buffer Loss");
        REQUIRE(it->format == "progressbar");
        REQUIRE(it->help.find("Excludes samples lost inside the device or driver") != std::string::npos);
        return std::any_cast<std::pair<std::string, F32>>(it->value);
    };
    REQUIRE(metric() == expected);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(metric() == expected);

    Parser::Map update;
    update["frequency"] = 100.0e6f;
    REQUIRE(flowgraph->blockReconfigure("radio", update) == Result::SUCCESS);
    REQUIRE(metric() == expected);

    testSoapyState.reads = std::make_shared<TestSoapyReads>();
    update.clear();
    update["bufferMultiplier"] = config.bufferMultiplier + 1;
    REQUIRE(flowgraph->blockReconfigure("radio", update) == Result::SUCCESS);
    REQUIRE(metric() == std::pair<std::string, F32>{"0.00%", 0.0f});
}
