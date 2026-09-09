#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <any>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
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
#include "jetstream/memory/axis.hh"
#include "jetstream/module_context.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "jetstream/scheduler_context.hh"
#include "module_impl.hh"
#include "soapysdr.hh"

using namespace Jetstream;

namespace {

constexpr const char* TestSoapyDriver = "cyberether_test";

struct TestSoapyReads {
    std::vector<int> results;
    std::vector<int> flags;
    std::vector<long long> timestamps;
    std::atomic<size_t> calls{0};
    std::atomic<bool> inFlight{false};
    bool throwOnRead = false;
    bool throwUnknown = false;
    void* lastBuffer = nullptr;
    size_t lastReadSize = 0;
    long lastTimeoutUs = 0;
    bool metadataReset = true;
};

struct TestSoapyState {
    bool advertiseBiasTee = true;
    bool throwOnSettingInfo = false;
    bool failStreamSetup = false;
    std::string failAt;
    bool throwUnknown = false;
    int activationResult = 0;
    std::vector<std::string> lifecycle;
    int streamDirection = -1;
    std::string streamFormat;
    std::vector<size_t> streamChannels;
    SoapySDR::Kwargs streamArgs;
    bool releasedWhileReading = false;
    std::vector<std::string> biasTeeWrites;
    std::shared_ptr<TestSoapyReads> reads = std::make_shared<TestSoapyReads>();
};

TestSoapyState testSoapyState;

void RecordSoapyCall(const std::string& call) {
    testSoapyState.lifecycle.push_back(call);
    if (testSoapyState.failAt == call) {
        if (testSoapyState.throwUnknown) {
            throw -1;
        }
        throw std::runtime_error("test failure: " + call);
    }
}

class TestSoapyDevice final : public SoapySDR::Device {
 public:
    ~TestSoapyDevice() override {
        testSoapyState.releasedWhileReading |= reads->inFlight.load();
        testSoapyState.lifecycle.push_back("unmake");
    }

    SoapySDR::RangeList getSampleRateRange(const int, const size_t) const override {
        RecordSoapyCall("sampleRateRanges");
        return {SoapySDR::Range(1.0, 10.0e6)};
    }

    SoapySDR::RangeList getFrequencyRange(const int, const size_t) const override {
        RecordSoapyCall("frequencyRanges");
        return {SoapySDR::Range(1.0, 2.0e9)};
    }

    SoapySDR::ArgInfoList getSettingInfo() const override {
        RecordSoapyCall("settings");
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
            RecordSoapyCall("biastee:" + value);
        }
    }

    void setSampleRate(const int, const size_t, const double) override {
        RecordSoapyCall("sampleRate");
    }

    void setFrequency(const int, const size_t, const double,
                      const SoapySDR::Kwargs&) override {
        RecordSoapyCall("frequency");
    }

    void setGainMode(const int, const size_t, const bool) override {
        RecordSoapyCall("gainMode");
    }

    SoapySDR::Stream* setupStream(const int direction,
                                  const std::string& format,
                                  const std::vector<size_t>& channels,
                                  const SoapySDR::Kwargs& args) override {
        RecordSoapyCall("setup");
        testSoapyState.streamDirection = direction;
        testSoapyState.streamFormat = format;
        testSoapyState.streamChannels = channels;
        testSoapyState.streamArgs = args;
        if (testSoapyState.failStreamSetup) {
            return nullptr;
        }
        return reinterpret_cast<SoapySDR::Stream*>(this);
    }

    int activateStream(SoapySDR::Stream*, const int, const long long,
                       const size_t) override {
        RecordSoapyCall("activate");
        return testSoapyState.activationResult;
    }

    int deactivateStream(SoapySDR::Stream*, const int, const long long) override {
        testSoapyState.releasedWhileReading |= reads->inFlight.load();
        RecordSoapyCall("deactivate");
        return 0;
    }

    void closeStream(SoapySDR::Stream*) override {
        testSoapyState.releasedWhileReading |= reads->inFlight.load();
        RecordSoapyCall("close");
    }

    int readStream(SoapySDR::Stream*,
                   void* const* buffers,
                   const size_t numElems,
                   int& flags,
                   long long& timeNs,
                   const long timeoutUs) override {
        struct ReadScope {
            std::atomic<bool>& inFlight;
            ~ReadScope() { inFlight = false; }
        } scope{reads->inFlight};
        reads->inFlight = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const auto index = reads->calls.fetch_add(1);
        reads->lastBuffer = buffers[0];
        reads->lastReadSize = numElems;
        reads->lastTimeoutUs = timeoutUs;
        reads->metadataReset &= flags == 0 && timeNs == 0;
        flags = index < reads->flags.size() ? reads->flags[index] : 0;
        timeNs = index < reads->timestamps.size() ? reads->timestamps[index] : 0;
        if (reads->throwOnRead) {
            if (reads->throwUnknown) {
                throw -1;
            }
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
    RecordSoapyCall("make");
    return new TestSoapyDevice();
}

const SoapySDR::Registry testSoapyRegistry(TestSoapyDriver,
                                           FindTestSoapyDevice,
                                           MakeTestSoapyDevice,
                                           SOAPY_SDR_ABI_VERSION);

constexpr const char* TestDiscoveryDriver = "cyberether_discovery_test";

struct TestDiscoveryState {
    std::mutex mutex;
    std::condition_variable changed;
    SoapySDR::KwargsList entries;
    std::vector<SoapySDR::Kwargs> queries;
    SoapySDR::Kwargs lastMakeArgs;
    size_t makes = 0;
    size_t blockCall = 0;
    bool blocked = false;
    bool released = false;
};

TestDiscoveryState testDiscoveryState;

SoapySDR::KwargsList FindDiscoveryDevice(const SoapySDR::Kwargs& args) {
    std::unique_lock lock(testDiscoveryState.mutex);
    testDiscoveryState.queries.push_back(args);
    SoapySDR::KwargsList entries;
    for (const auto& entry : testDiscoveryState.entries) {
        if (!args.contains("serial") || args.at("serial") == entry.at("serial")) {
            entries.push_back(entry);
        }
    }
    if (testDiscoveryState.queries.size() == testDiscoveryState.blockCall) {
        testDiscoveryState.blocked = true;
        testDiscoveryState.changed.notify_all();
        testDiscoveryState.changed.wait(lock, [] { return testDiscoveryState.released; });
    }
    return entries;
}

SoapySDR::Device* MakeDiscoveryDevice(const SoapySDR::Kwargs& args) {
    {
        std::lock_guard lock(testDiscoveryState.mutex);
        ++testDiscoveryState.makes;
        testDiscoveryState.lastMakeArgs = args;
        RecordSoapyCall("make");
        if (std::none_of(testDiscoveryState.entries.begin(), testDiscoveryState.entries.end(),
                        [&](const auto& entry) { return entry.at("serial") == args.at("serial"); })) {
            throw std::runtime_error("test discovery device unavailable");
        }
    }
    return new TestSoapyDevice();
}

const SoapySDR::Registry testDiscoveryRegistry(TestDiscoveryDriver,
                                              FindDiscoveryDevice,
                                              MakeDiscoveryDevice,
                                              SOAPY_SDR_ABI_VERSION);

struct SoapyDiscoveryFixture {
    using Devices = Modules::SoapyDiscovery::DeviceList;

    const std::string filter = std::string("driver=") + TestDiscoveryDriver;
    const SoapySDR::Kwargs deviceA{{"label", "Discovery A"}, {"serial", "A"}};
    const SoapySDR::Kwargs deviceB{{"label", "Discovery B"}, {"serial", "B"}};
    std::vector<std::future<Devices>> requests;

    SoapyDiscoveryFixture() {
        Modules::SoapyDiscovery::ClearDiscoveryCache();
        testSoapyState = {};
        std::lock_guard lock(testDiscoveryState.mutex);
        testDiscoveryState.entries = {deviceA, deviceB};
        testDiscoveryState.queries.clear();
        testDiscoveryState.lastMakeArgs.clear();
        testDiscoveryState.makes = 0;
        testDiscoveryState.blockCall = 0;
        testDiscoveryState.blocked = false;
        testDiscoveryState.released = false;
    }

    ~SoapyDiscoveryFixture() {
        release();
        for (auto& request : requests) {
            if (request.valid()) {
                request.wait();
            }
        }
        Modules::SoapyDiscovery::ClearDiscoveryCache();
        std::lock_guard lock(testDiscoveryState.mutex);
        testDiscoveryState.entries.clear();
    }

    void setEntries(SoapySDR::KwargsList entries) {
        std::lock_guard lock(testDiscoveryState.mutex);
        testDiscoveryState.entries = std::move(entries);
    }

    size_t queryCount() const {
        std::lock_guard lock(testDiscoveryState.mutex);
        return testDiscoveryState.queries.size();
    }

    size_t makeCount() const {
        std::lock_guard lock(testDiscoveryState.mutex);
        return testDiscoveryState.makes;
    }

    size_t request(const std::string& args) {
        const auto index = requests.size();
        requests.push_back(std::async(std::launch::async, [args] {
            return Modules::SoapyDiscovery::ListDevices(args);
        }));
        return index;
    }

    void blockFirstQuery() {
        std::lock_guard lock(testDiscoveryState.mutex);
        testDiscoveryState.blockCall = 1;
    }

    bool waitUntilBlocked() {
        std::unique_lock lock(testDiscoveryState.mutex);
        return testDiscoveryState.changed.wait_for(lock, std::chrono::seconds(5), [] {
            return testDiscoveryState.blocked;
        });
    }

    void release() {
        {
            std::lock_guard lock(testDiscoveryState.mutex);
            testDiscoveryState.released = true;
        }
        testDiscoveryState.changed.notify_all();
    }
};

struct SoapySelectionFixture : SoapyDiscoveryFixture, FlowgraphFixture {};

struct SoapyImplAccess : Modules::SoapyImpl {
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
        const SoapyModuleCleanup cleanup{module};
        REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        const auto writesBeforeDestroy = testSoapyState.biasTeeWrites;

        REQUIRE(module->destroy() == Result::SUCCESS);
        REQUIRE(writesBeforeDestroy == std::vector<std::string>{"true"});
        REQUIRE(testSoapyState.biasTeeWrites ==
                std::vector<std::string>{"true", "false"});
    }

    SECTION("disabled antenna power is not written again on shutdown") {
        auto disabledConfig = config;
        disabledConfig.biasTee = false;
        const auto module = BuildTestSoapyModule();
        const SoapyModuleCleanup cleanup{module};
        REQUIRE(module->create("test", disabledConfig, {}) == Result::SUCCESS);
        REQUIRE(module->destroy() == Result::SUCCESS);
        REQUIRE(testSoapyState.biasTeeWrites == std::vector<std::string>{"false"});
    }

    SECTION("an explicit disable removes the shutdown cleanup obligation") {
        const auto module = BuildTestSoapyModule();
        const SoapyModuleCleanup cleanup{module};
        REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        REQUIRE(module->getImpl<Modules::SoapyImpl>()->setBiasTee(false) == Result::SUCCESS);
        REQUIRE(module->destroy() == Result::SUCCESS);
        REQUIRE(testSoapyState.biasTeeWrites ==
                std::vector<std::string>{"true", "false"});
    }

    SECTION("a failed explicit disable is retried during shutdown") {
        const auto module = BuildTestSoapyModule();
        const SoapyModuleCleanup cleanup{module};
        REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        testSoapyState.failAt = "biastee:false";
        REQUIRE(module->getImpl<Modules::SoapyImpl>()->setBiasTee(false) == Result::ERROR);
        testSoapyState.failAt.clear();
        REQUIRE(module->destroy() == Result::SUCCESS);
        REQUIRE(testSoapyState.biasTeeWrites ==
                std::vector<std::string>{"true", "false", "false"});
    }

    SECTION("capabilities are refreshed when the module is recreated") {
        const auto module = BuildTestSoapyModule();
        const SoapyModuleCleanup cleanup{module};
        REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        REQUIRE(module->destroy() == Result::SUCCESS);
        testSoapyState.biasTeeWrites.clear();

        SECTION("new device does not advertise Bias-T") {
            testSoapyState.advertiseBiasTee = false;
        }
        SECTION("new device cannot report optional settings") {
            testSoapyState.throwOnSettingInfo = true;
        }

        REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        REQUIRE(module->getImpl<Modules::SoapyImpl>()->setBiasTee(false) == Result::SUCCESS);
        REQUIRE(module->destroy() == Result::SUCCESS);
        REQUIRE(testSoapyState.biasTeeWrites.empty());
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
        const SoapyModuleCleanup cleanup{module};
        REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        REQUIRE(module->destroy() == Result::SUCCESS);
        REQUIRE(testSoapyState.biasTeeWrites.empty());
    }

    SECTION("unadvertised Bias-T is not written") {
        testSoapyState.advertiseBiasTee = false;

        const auto module = BuildTestSoapyModule();
        const SoapyModuleCleanup cleanup{module};
        REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        REQUIRE(module->destroy() == Result::SUCCESS);
        REQUIRE(testSoapyState.biasTeeWrites.empty());
    }
}

TEST_CASE("Soapy lifecycle preserves stream configuration and output layout",
          "[modules][soapy][devices][lifecycle]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    testSoapyState = {};
    auto config = TestDeviceSoapyConfig();
    config.numberOfBatches = 3;
    config.numberOfTimeSamples = 17;
    config.bufferMultiplier = 2;
    config.streamString = "bufflen=4096,buffers=3";
    const auto module = BuildTestSoapyModule();
    const SoapyModuleCleanup cleanup{module};
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);

    REQUIRE(testSoapyState.lifecycle == std::vector<std::string>{
        "make", "sampleRateRanges", "frequencyRanges", "settings",
        "sampleRate", "frequency", "gainMode", "biastee:true", "setup", "activate",
    });
    REQUIRE(testSoapyState.streamDirection == SOAPY_SDR_RX);
    REQUIRE(testSoapyState.streamFormat == "CF32");
    REQUIRE(testSoapyState.streamChannels == std::vector<size_t>{0});
    REQUIRE(testSoapyState.streamArgs == SoapySDR::Kwargs{
        {"bufflen", "4096"}, {"buffers", "3"},
    });

    const auto& output = module->outputs().at("signal").tensor;
    REQUIRE(output.dtype() == DataType::CF32);
    REQUIRE(output.shape() == Shape{3, 17});
    REQUIRE(std::any_cast<F32>(output.attribute("frequency")) == config.frequency);
    REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == config.sampleRate);
    SignalAxes axes;
    REQUIRE(ResolveSignalAxes(output, axes) == Result::SUCCESS);
    REQUIRE(axes.sample == Index{1});
    REQUIRE(axes.batch == Index{0});
    REQUIRE_FALSE(axes.channel.has_value());

    auto* soapy = module->getImpl<Modules::SoapyImpl>();
    REQUIRE((soapy->*SoapyImplAccess::circularBufferMember()).capacity() == 102);

    testSoapyState.lifecycle.clear();
    Parser::Map update;
    update["frequency"] = 100.0e6f;
    update["sampleRate"] = 1.5e6f;
    update["automaticGain"] = false;
    REQUIRE(module->reconfigure(update) == Result::SUCCESS);
    REQUIRE(testSoapyState.lifecycle == std::vector<std::string>{
        "frequency", "sampleRate", "gainMode",
    });
    REQUIRE(std::any_cast<F32>(output.attribute("frequency")) == 100.0e6f);
    REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == 1.5e6f);

    testSoapyState.lifecycle.clear();
    REQUIRE(WaitForSoapy([&] { return testSoapyState.reads->inFlight.load(); }));
    REQUIRE(module->destroy() == Result::SUCCESS);
    REQUIRE(testSoapyState.lifecycle == std::vector<std::string>{
        "deactivate", "close", "biastee:false", "unmake",
    });
    REQUIRE_FALSE(testSoapyState.releasedWhileReading);
    REQUIRE_FALSE(testSoapyState.reads->inFlight.load());
    REQUIRE(soapy->getBufferHealth() == 0.0f);
    REQUIRE(soapy->getThroughput() == std::pair<F32, F32>{0.0f, 0.0f});

    const auto calls = testSoapyState.lifecycle;
    REQUIRE(soapy->destroy() == Result::SUCCESS);
    REQUIRE(testSoapyState.lifecycle == calls);
}

TEST_CASE("Soapy implementation scope stops reception before releasing hardware",
          "[modules][soapy][devices][lifecycle]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    testSoapyState = {};
    {
        Modules::SoapyImpl impl;
        const auto handle = std::shared_ptr<Modules::SoapyImpl>(&impl, [](auto*) {});
        Module module(DeviceType::CPU, RuntimeType::NATIVE, "generic",
                      handle, nullptr, handle, impl.candidate());
        REQUIRE(module.create("test", TestDeviceSoapyConfig(), {}) == Result::SUCCESS);
        REQUIRE(WaitForSoapy([&] { return testSoapyState.reads->inFlight.load(); }));
        testSoapyState.lifecycle.clear();
    }
    REQUIRE(testSoapyState.lifecycle == std::vector<std::string>{
        "deactivate", "close", "biastee:false", "unmake",
    });
    REQUIRE_FALSE(testSoapyState.releasedWhileReading);
    REQUIRE_FALSE(testSoapyState.reads->inFlight.load());
}

TEST_CASE("Soapy missing devices remain incomplete without acquiring resources",
          "[modules][soapy][devices][lifecycle]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    testSoapyState = {};
    auto config = TestDeviceSoapyConfig();
    config.deviceString = "driver=cyberether_missing_test_driver";
    const auto module = BuildTestSoapyModule();
    REQUIRE(module->create("test", config, {}) == Result::INCOMPLETE);
    REQUIRE(module->state() == Module::State::INCOMPLETE);
    REQUIRE(module->outputs().empty());
    REQUIRE(testSoapyState.lifecycle.empty());
    REQUIRE(module->destroy() == Result::SUCCESS);
    REQUIRE(testSoapyState.lifecycle.empty());

    REQUIRE(module->create("test", TestDeviceSoapyConfig(), {}) == Result::SUCCESS);
    const SoapyModuleCleanup cleanup{module};
    REQUIRE(module->destroy() == Result::SUCCESS);
    REQUIRE_FALSE(testSoapyState.releasedWhileReading);
}

TEST_CASE("Soapy creation failures release only acquired resources",
          "[modules][soapy][devices][lifecycle]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    for (const std::string failure : {"make", "sampleRateRanges", "frequencyRanges",
                                      "sampleRate", "frequency", "gainMode",
                                      "biastee:true", "setup", "activate"}) {
        for (const bool unknown : {false, true}) {
            DYNAMIC_SECTION("Failure: " << failure << " Unknown exception: " << unknown) {
                testSoapyState = {};
                testSoapyState.failAt = failure;
                testSoapyState.throwUnknown = unknown;
                const auto module = BuildTestSoapyModule();
                const SoapyModuleCleanup cleanup{module};
                REQUIRE(module->create("test", TestDeviceSoapyConfig(), {}) == Result::ERROR);
                REQUIRE(module->state() == Module::State::DESTROYED);
                const auto& calls = testSoapyState.lifecycle;
                REQUIRE(std::count(calls.begin(), calls.end(), "unmake") ==
                        (failure == "make" ? 0 : 1));
                REQUIRE(std::count(calls.begin(), calls.end(), "close") ==
                        (failure == "activate" ? 1 : 0));
                REQUIRE(std::count(calls.begin(), calls.end(), "deactivate") == 0);
                REQUIRE(testSoapyState.reads->calls == 0);
                if (failure == "biastee:true" || failure == "setup" || failure == "activate") {
                    REQUIRE(testSoapyState.biasTeeWrites ==
                            std::vector<std::string>{"true", "false"});
                    REQUIRE(calls[calls.size() - 2] == "biastee:false");
                    REQUIRE(calls.back() == "unmake");
                } else {
                    REQUIRE(testSoapyState.biasTeeWrites.empty());
                }

                testSoapyState.failAt.clear();
                REQUIRE(module->create("test", TestDeviceSoapyConfig(), {}) == Result::SUCCESS);
                REQUIRE(module->destroy() == Result::SUCCESS);
                REQUIRE_FALSE(testSoapyState.releasedWhileReading);
            }
        }
    }
}

TEST_CASE("Soapy rejected capabilities and stream statuses clean up before returning",
          "[modules][soapy][devices][lifecycle]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    testSoapyState = {};
    auto config = TestDeviceSoapyConfig();
    std::vector<std::string> expectedBiasTeeWrites;
    int expectedCloses = 0;

    SECTION("unsupported sample rate") {
        config.sampleRate = 20.0e6f;
    }
    SECTION("unsupported frequency") {
        config.frequency = 3.0e9f;
    }
    SECTION("null stream") {
        testSoapyState.failStreamSetup = true;
        expectedBiasTeeWrites = {"true", "false"};
    }
    SECTION("activation error status") {
        testSoapyState.activationResult = SOAPY_SDR_STREAM_ERROR;
        expectedBiasTeeWrites = {"true", "false"};
        expectedCloses = 1;
    }

    const auto module = BuildTestSoapyModule();
    const SoapyModuleCleanup cleanup{module};
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    const auto& calls = testSoapyState.lifecycle;
    REQUIRE(std::count(calls.begin(), calls.end(), "unmake") == 1);
    REQUIRE(std::count(calls.begin(), calls.end(), "close") == expectedCloses);
    REQUIRE(std::count(calls.begin(), calls.end(), "deactivate") == 0);
    REQUIRE(testSoapyState.biasTeeWrites == expectedBiasTeeWrites);
    REQUIRE(testSoapyState.reads->calls == 0);
}

TEST_CASE("Soapy teardown contains driver exceptions and still releases the device",
          "[modules][soapy][devices][bias-tee][lifecycle]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    for (const std::string failure : {"deactivate", "close", "biastee:false"}) {
        for (const bool unknown : {false, true}) {
            DYNAMIC_SECTION("Failure: " << failure << " Unknown exception: " << unknown) {
                testSoapyState = {};
                const auto module = BuildTestSoapyModule();
                const SoapyModuleCleanup cleanup{module};
                REQUIRE(module->create("test", TestDeviceSoapyConfig(), {}) == Result::SUCCESS);
                testSoapyState.failAt = failure;
                testSoapyState.throwUnknown = unknown;
                REQUIRE(module->destroy() == Result::SUCCESS);
                const auto calls = testSoapyState.lifecycle;
                REQUIRE(std::count(calls.begin(), calls.end(), "unmake") == 1);
                REQUIRE(testSoapyState.biasTeeWrites ==
                        std::vector<std::string>{"true", "false"});
                REQUIRE_FALSE(testSoapyState.releasedWhileReading);
                REQUIRE(module->getImpl<Modules::SoapyImpl>()->destroy() == Result::SUCCESS);
                REQUIRE(testSoapyState.lifecycle == calls);
            }
        }
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
            testSoapyState = {};
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("soapy",
                                          implementation.device,
                                          implementation.runtime,
                                          implementation.provider,
                                          module) == Result::SUCCESS);
            const SoapyModuleCleanup cleanup{module};
            REQUIRE(module->create("test", TestDeviceSoapyConfig(), {}) == Result::SUCCESS);

            auto* soapy = module->getImpl<Modules::SoapyImpl>();
            REQUIRE(soapy != nullptr);

            auto& candidate = *soapy->candidate();
            candidate.sampleRate = 20.0e6f;
            candidate.frequency = 3.0e9f;

            const auto calls = testSoapyState.lifecycle;
            REQUIRE(soapy->setSampleRate(candidate.sampleRate) == Result::WARNING);
            REQUIRE(soapy->setTunerFrequency(candidate.frequency) == Result::WARNING);
            REQUIRE(testSoapyState.lifecycle == calls);
            REQUIRE(soapy->validate() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Soapy device controls reject changes while closed",
          "[modules][soapy][devices][lifecycle]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    testSoapyState = {};
    const auto module = BuildTestSoapyModule();
    const SoapyModuleCleanup cleanup{module};
    auto* soapy = module->getImpl<Modules::SoapyImpl>();
    const auto requireClosed = [&] {
        const auto calls = testSoapyState.lifecycle;
        REQUIRE(soapy->setTunerFrequency(100.0e6f) == Result::ERROR);
        REQUIRE(soapy->setSampleRate(1.5e6f) == Result::ERROR);
        REQUIRE(soapy->setAutomaticGain(false) == Result::ERROR);
        REQUIRE(soapy->setBiasTee(true) == Result::ERROR);
        REQUIRE(testSoapyState.lifecycle == calls);
    };

    requireClosed();
    REQUIRE(module->create("test", TestDeviceSoapyConfig(), {}) == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
    requireClosed();
}

TEST_CASE("Soapy failed device controls preserve module configuration and metadata",
          "[modules][soapy][devices][lifecycle]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    for (const std::string failure : {"frequency", "sampleRate", "gainMode"}) {
        DYNAMIC_SECTION("Failed control: " << failure) {
            testSoapyState = {};
            const auto config = TestDeviceSoapyConfig();
            const auto module = BuildTestSoapyModule();
            const SoapyModuleCleanup cleanup{module};
            REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
            auto* soapy = module->getImpl<Modules::SoapyImpl>();
            testSoapyState.failAt = failure;

            if (failure == "frequency") {
                REQUIRE(soapy->setTunerFrequency(100.0e6f) == Result::ERROR);
            } else if (failure == "sampleRate") {
                REQUIRE(soapy->setSampleRate(1.5e6f) == Result::ERROR);
            } else {
                REQUIRE(soapy->setAutomaticGain(false) == Result::ERROR);
            }

            const auto& applied = static_cast<const Modules::Soapy&>(module->config());
            REQUIRE(applied.frequency == config.frequency);
            REQUIRE(applied.sampleRate == config.sampleRate);
            REQUIRE(applied.automaticGain == config.automaticGain);
            const auto& output = module->outputs().at("signal").tensor;
            REQUIRE(std::any_cast<F32>(output.attribute("frequency")) == config.frequency);
            REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == config.sampleRate);
            REQUIRE(module->destroy() == Result::SUCCESS);
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

    const auto devices = Modules::SoapyDiscovery::BuildDeviceList(entries);

    REQUIRE(devices.size() == entries.size());
    REQUIRE(devices.at("RTL-SDR").at("serial") == "A");
    REQUIRE(devices.at("RTL-SDR [B]").at("serial") == "B");
    REQUIRE(devices.at("RTL-SDR [B] #2").at("serial") == "B");
    REQUIRE(devices.contains("remote"));
    REQUIRE(devices.contains("remote #2"));
}

TEST_CASE_METHOD(SoapyDiscoveryFixture,
                 "Soapy discovery caches parsed filters and returns independent results",
                 "[modules][soapy][discovery]") {
    auto first = Modules::SoapyDiscovery::ListDevices(filter + ",serial=A,remote=host-a");
    REQUIRE(first.size() == 1);
    REQUIRE(first.at("Discovery A").at("serial") == "A");
    first.at("Discovery A")["serial"] = "modified";

    const auto repeated = Modules::SoapyDiscovery::ListDevices(
        " remote = host-a, serial = A, " + filter);
    REQUIRE(repeated.at("Discovery A").at("serial") == "A");
    REQUIRE(queryCount() == 1);
    REQUIRE(makeCount() == 0);
    {
        std::lock_guard lock(testDiscoveryState.mutex);
        REQUIRE(testDiscoveryState.queries.front() == SoapySDR::Kwargs{
            {"driver", TestDiscoveryDriver}, {"serial", "A"}, {"remote", "host-a"},
        });
    }
}

TEST_CASE_METHOD(SoapyDiscoveryFixture,
                 "Soapy discovery isolates device and remote filters",
                 "[modules][soapy][discovery]") {
    const auto first = Modules::SoapyDiscovery::ListDevices(filter + ",serial=A,remote=host-a");
    const auto second = Modules::SoapyDiscovery::ListDevices(filter + ",serial=B,remote=host-a");
    const auto remote = Modules::SoapyDiscovery::ListDevices(filter + ",serial=A,remote=host-b");
    REQUIRE(first.at("Discovery A").at("serial") == "A");
    REQUIRE(second.at("Discovery B").at("serial") == "B");
    REQUIRE(remote.at("Discovery A").at("serial") == "A");
    REQUIRE(queryCount() == 3);
    REQUIRE(makeCount() == 0);
}

TEST_CASE_METHOD(SoapyDiscoveryFixture,
                 "Soapy discovery supports forced refresh and invalidation",
                 "[modules][soapy][discovery]") {
    const auto initial = Modules::SoapyDiscovery::ListDevices(filter);
    REQUIRE(initial.size() == 2);
    setEntries({deviceB});
    REQUIRE(Modules::SoapyDiscovery::ListDevices(filter) == initial);
    REQUIRE(queryCount() == 1);

    const auto refreshed = Modules::SoapyDiscovery::ListDevices(filter, true);
    REQUIRE(refreshed.size() == 1);
    REQUIRE(refreshed.contains("Discovery B"));
    REQUIRE(Modules::SoapyDiscovery::ListDevices(filter) == refreshed);
    REQUIRE(queryCount() == 2);

    REQUIRE(Modules::SoapyDiscovery::LoadDriverLibrary("") == Result::SUCCESS);
    REQUIRE(Modules::SoapyDiscovery::ListDevices(filter) == refreshed);
    REQUIRE(queryCount() == 2);

    setEntries({deviceA});
    Modules::SoapyDiscovery::ClearDiscoveryCache();
    const auto invalidated = Modules::SoapyDiscovery::ListDevices(filter);
    REQUIRE(invalidated.size() == 1);
    REQUIRE(invalidated.contains("Discovery A"));
    REQUIRE(queryCount() == 3);
    REQUIRE(makeCount() == 0);
}

TEST_CASE_METHOD(SoapyDiscoveryFixture,
                 "Soapy discovery expires populated and empty snapshots",
                 "[modules][soapy][discovery]") {
    using namespace std::chrono_literals;
    const auto start = Modules::SoapyDiscovery::Clock::time_point{};

    SECTION("populated snapshot") {}
    SECTION("empty snapshot") {
        setEntries({});
    }

    const auto initial = Modules::SoapyDiscovery::ListDevices(filter, false, start);
    setEntries({deviceB});
    REQUIRE(Modules::SoapyDiscovery::ListDevices(filter, false, start + 999ms) == initial);
    REQUIRE(queryCount() == 1);
    const auto expired = Modules::SoapyDiscovery::ListDevices(filter, false, start + 1s);
    REQUIRE(expired.size() == 1);
    REQUIRE(expired.contains("Discovery B"));
    REQUIRE(queryCount() == 2);
}

TEST_CASE_METHOD(SoapyDiscoveryFixture,
                 "Soapy discovery shares in-flight queries without blocking other filters",
                 "[modules][soapy][discovery][concurrency]") {
    blockFirstQuery();
    const auto first = request(filter);
    REQUIRE(waitUntilBlocked());
    std::vector<size_t> followers;
    for (int i = 0; i < 6; ++i) {
        followers.push_back(request(filter));
    }

    const auto other = request(filter + ",serial=B");
    REQUIRE(requests[other].wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    REQUIRE(requests[other].get().size() == 1);
    release();
    const auto initial = requests[first].get();
    REQUIRE(initial.size() == 2);
    for (const auto index : followers) {
        REQUIRE(requests[index].get() == initial);
    }
    REQUIRE(queryCount() == 2);
    REQUIRE(makeCount() == 0);
}

TEST_CASE_METHOD(SoapyDiscoveryFixture,
                 "Soapy discovery invalidation prevents stale in-flight results from being cached",
                 "[modules][soapy][discovery][concurrency]") {
    blockFirstQuery();
    const auto first = request(filter);
    REQUIRE(waitUntilBlocked());
    setEntries({deviceB});
    Modules::SoapyDiscovery::ClearDiscoveryCache();

    const auto fresh = request(filter);
    REQUIRE(requests[fresh].wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    const auto refreshed = requests[fresh].get();
    REQUIRE(refreshed.size() == 1);
    REQUIRE(refreshed.contains("Discovery B"));
    release();
    REQUIRE(requests[first].get().size() == 2);
    REQUIRE(Modules::SoapyDiscovery::ListDevices(filter) == refreshed);
    REQUIRE(queryCount() == 2);
}

TEST_CASE_METHOD(SoapyDiscoveryFixture,
                 "Soapy cached disconnected devices become incomplete after opening fails",
                 "[modules][soapy][discovery][lifecycle]") {
    const auto selected = filter + ",serial=A";
    REQUIRE(Modules::SoapyDiscovery::ListDevices(selected).size() == 1);
    setEntries({deviceB});

    Modules::SoapyReceiver receiver;
    REQUIRE(receiver.open(SoapySDR::KwargsFromString(selected)) == Result::INCOMPLETE);
    REQUIRE(makeCount() == 1);
    REQUIRE(Modules::SoapyDiscovery::ListDevices(selected).empty());
    REQUIRE(queryCount() == 3);
}

TEST_CASE_METHOD(SoapySelectionFixture,
                 "Soapy blocks classify opening failures using fresh selection results",
                 "[modules][soapy][block][selection][lifecycle]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    Blocks::Soapy config;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    REQUIRE(flowgraph->blockCreate("radio", config, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Incomplete);

    const auto selector = filter + ",serial=A,remote=host-a";
    REQUIRE(Modules::SoapyDiscovery::ListDevices(selector).size() == 1);
    testSoapyState.failAt = "make";

    Block::State expected = Block::State::Incomplete;
    std::string diagnostic;
    SECTION("the selected device was disconnected") {
        setEntries({deviceB});
        diagnostic = "no longer available";
    }
    SECTION("the selection became ambiguous") {
        setEntries({deviceA, deviceA});
        diagnostic = "ambiguous";
    }
    SECTION("a uniquely available device rejects initialization") {
        diagnostic = "test failure: make";
        expected = Block::State::Errored;
    }

    REQUIRE(flowgraph->blockReconfigure("radio", {{"deviceString", selector}}) == Result::SUCCESS);
    const auto block = viewBlock("radio");
    REQUIRE(block.state == expected);
    REQUIRE(block.outputs.empty());
    REQUIRE(block.diagnostic.find(diagnostic) != std::string::npos);
    REQUIRE(Parser::Get<std::string>(block.config, "deviceString") == selector);
    REQUIRE(makeCount() == 1);
    {
        std::lock_guard lock(testDiscoveryState.mutex);
        REQUIRE(testDiscoveryState.queries.back() == SoapySDR::KwargsFromString(selector));
        REQUIRE(testDiscoveryState.lastMakeArgs.at("serial") == "A");
        REQUIRE(testDiscoveryState.lastMakeArgs.at("remote") == "host-a");
    }

    testSoapyState.failAt.clear();
    setEntries({deviceA, deviceB});
    Modules::SoapyDiscovery::ClearDiscoveryCache();
    REQUIRE(flowgraph->blockRecreate("radio", block.config) == Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Created);
    REQUIRE(Parser::Get<std::string>(viewBlock("radio").config, "deviceString") == selector);
    REQUIRE(makeCount() == 2);
}

TEST_CASE("Soapy typed reads require an active stream and a nonempty buffer",
          "[modules][soapy][receive][lifecycle]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    testSoapyState = {};
    Modules::SoapyReceiver receiver;
    std::array<CF32, 8> samples;
    samples.fill(CF32{7.0f, -7.0f});
    auto target = std::span<CF32>{samples};
    std::string diagnostic = "active stream";
    const SoapySDR::Kwargs args{{"driver", TestSoapyDriver}};

    SECTION("closed device") {}
    SECTION("open device without a stream") {
        REQUIRE(receiver.open(args) == Result::SUCCESS);
    }
    SECTION("failed stream setup") {
        REQUIRE(receiver.open(args) == Result::SUCCESS);
        testSoapyState.failStreamSetup = true;
        REQUIRE(receiver.startStream({}) == Result::ERROR);
    }
    SECTION("stream whose activation failed") {
        REQUIRE(receiver.open(args) == Result::SUCCESS);
        testSoapyState.activationResult = SOAPY_SDR_STREAM_ERROR;
        REQUIRE(receiver.startStream({}) == Result::ERROR);
    }
    SECTION("reset device") {
        REQUIRE(receiver.open(args) == Result::SUCCESS);
        REQUIRE(receiver.startStream({}) == Result::SUCCESS);
        receiver.reset();
    }
    SECTION("empty buffer with an active stream") {
        REQUIRE(receiver.open(args) == Result::SUCCESS);
        REQUIRE(receiver.startStream({}) == Result::SUCCESS);
        target = {};
        diagnostic = "empty";
    }

    const auto received = receiver.read(target);
    REQUIRE(received.status == Modules::SoapyReceiver::ReadStatus::Error);
    REQUIRE(received.sampleCount == 0);
    REQUIRE_FALSE(received.timestampNs.has_value());
    REQUIRE(received.error.find(diagnostic) != std::string::npos);
    REQUIRE(testSoapyState.reads->calls == 0);
    REQUIRE(std::all_of(samples.begin(), samples.end(), [](const auto sample) {
        return sample == CF32{7.0f, -7.0f};
    }));

    if (target.empty()) {
        testSoapyState.reads->results = {1};
        REQUIRE(receiver.read(samples).status == Modules::SoapyReceiver::ReadStatus::Samples);
    }
}

TEST_CASE("Soapy typed reads use the caller buffer and preserve only valid timestamps",
          "[modules][soapy][receive]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    testSoapyState = {};
    const auto reads = testSoapyState.reads;
    constexpr I64 timestamp = 1234567890123456789LL;
    reads->results = {2, 4, 4};
    reads->flags = {SOAPY_SDR_HAS_TIME, SOAPY_SDR_END_BURST, SOAPY_SDR_HAS_TIME};
    reads->timestamps = {timestamp, timestamp, 0};
    Modules::SoapyReceiver receiver;
    REQUIRE(receiver.open({{"driver", TestSoapyDriver}}) == Result::SUCCESS);
    REQUIRE(receiver.startStream({}) == Result::SUCCESS);

    std::array<CF32, 8> samples;
    samples.fill(CF32{7.0f, -7.0f});
    const auto target = std::span<CF32>{samples}.subspan(2, 4);
    const auto first = receiver.read(target);
    REQUIRE(first.status == Modules::SoapyReceiver::ReadStatus::Samples);
    REQUIRE(first.sampleCount == 2);
    REQUIRE(first.timestampNs == timestamp);
    REQUIRE(first.error.empty());
    REQUIRE(reads->lastBuffer == target.data());
    REQUIRE(reads->lastReadSize == target.size());
    REQUIRE(reads->lastTimeoutUs == 100000);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        REQUIRE(samples[i] == (i == 2 || i == 3 ? CF32{0.0f, -1.0f} : CF32{7.0f, -7.0f}));
    }

    const auto second = receiver.read(target);
    REQUIRE(second.status == Modules::SoapyReceiver::ReadStatus::Samples);
    REQUIRE(second.sampleCount == target.size());
    REQUIRE_FALSE(second.timestampNs.has_value());
    REQUIRE(second.error.empty());

    const auto third = receiver.read(target);
    REQUIRE(third.status == Modules::SoapyReceiver::ReadStatus::Samples);
    REQUIRE(third.timestampNs.has_value());
    REQUIRE(*third.timestampNs == 0);
    REQUIRE(reads->metadataReset);
    REQUIRE(reads->calls == 3);
}

TEST_CASE("Soapy typed reads translate idle, overflow, and failure results",
          "[modules][soapy][receive]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    using Status = Modules::SoapyReceiver::ReadStatus;
    for (const int result : {0, SOAPY_SDR_TIMEOUT, SOAPY_SDR_OVERFLOW,
                            SOAPY_SDR_STREAM_ERROR, SOAPY_SDR_CORRUPTION,
                            SOAPY_SDR_NOT_SUPPORTED, SOAPY_SDR_TIME_ERROR,
                            SOAPY_SDR_UNDERFLOW, -999, 9}) {
        DYNAMIC_SECTION("Driver result: " << result) {
            testSoapyState = {};
            const auto reads = testSoapyState.reads;
            reads->results = {result};
            reads->flags = {SOAPY_SDR_HAS_TIME};
            reads->timestamps = {1234};
            Modules::SoapyReceiver receiver;
            REQUIRE(receiver.open({{"driver", TestSoapyDriver}}) == Result::SUCCESS);
            REQUIRE(receiver.startStream({}) == Result::SUCCESS);
            std::array<CF32, 8> samples;
            const auto received = receiver.read(samples);

            REQUIRE(received.sampleCount == 0);
            REQUIRE_FALSE(received.timestampNs.has_value());
            REQUIRE(reads->calls == 1);
            if (result == 0 || result == SOAPY_SDR_TIMEOUT) {
                REQUIRE(received.status == Status::Timeout);
                REQUIRE(received.error.empty());
            } else if (result == SOAPY_SDR_OVERFLOW) {
                REQUIRE(received.status == Status::Overflow);
                REQUIRE(received.error.empty());
            } else {
                REQUIRE(received.status == Status::Error);
                if (result < 0) {
                    REQUIRE(received.error.find(SoapySDR::errToStr(result)) != std::string::npos);
                    REQUIRE(received.error.find("(" + std::to_string(result) + ")") != std::string::npos);
                } else {
                    REQUIRE(received.error.find("more samples than requested") != std::string::npos);
                }
            }
        }
    }
}

TEST_CASE("Soapy typed reads translate driver exceptions into failures",
          "[modules][soapy][receive]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    for (const bool unknown : {false, true}) {
        DYNAMIC_SECTION("Unknown exception: " << unknown) {
            testSoapyState = {};
            const auto reads = testSoapyState.reads;
            reads->throwOnRead = true;
            reads->throwUnknown = unknown;
            Modules::SoapyReceiver receiver;
            REQUIRE(receiver.open({{"driver", TestSoapyDriver}}) == Result::SUCCESS);
            REQUIRE(receiver.startStream({}) == Result::SUCCESS);
            std::array<CF32, 8> samples;
            const auto received = receiver.read(samples);

            REQUIRE(received.status == Modules::SoapyReceiver::ReadStatus::Error);
            REQUIRE(received.sampleCount == 0);
            REQUIRE_FALSE(received.timestampNs.has_value());
            REQUIRE(received.error == (unknown ? "Unknown driver exception" : "test receive failure"));
            REQUIRE(reads->calls == 1);
        }
    }
}

TEST_CASE("Soapy receive statuses distinguish idle reads from failures",
          "[modules][soapy][receive]") {
    using Status = Modules::SoapyReceiveStatus;
    using Action = Status::Action;
    using ReadStatus = Modules::SoapyReceiver::ReadStatus;
    const auto start = Status::Clock::time_point{};
    Status status;

    REQUIRE(status.handle(ReadStatus::Samples, start) == Action::Samples);
    REQUIRE(status.handle(ReadStatus::Timeout, start) == Action::Retry);
    REQUIRE(status.deviceOverflows == 0);

    REQUIRE(status.handle(ReadStatus::Error, start) == Action::Fail);
    REQUIRE(status.deviceOverflows == 0);
}

TEST_CASE("Soapy device overflow warnings are counted and rate limited per stream",
          "[modules][soapy][receive]") {
    using Status = Modules::SoapyReceiveStatus;
    using Action = Status::Action;
    using ReadStatus = Modules::SoapyReceiver::ReadStatus;
    using namespace std::chrono_literals;
    const auto start = Status::Clock::time_point{};
    Status status;

    REQUIRE(status.handle(ReadStatus::Overflow, start) == Action::WarnOverflow);
    REQUIRE(status.deviceOverflows == 1);
    for (int i = 0; i < 100; ++i) {
        REQUIRE(status.handle(ReadStatus::Overflow, start + 999ms) == Action::Retry);
    }
    REQUIRE(status.deviceOverflows == 101);

    REQUIRE(status.handle(ReadStatus::Samples, start + 999ms) == Action::Samples);
    REQUIRE(status.handle(ReadStatus::Timeout, start + 999ms) == Action::Retry);
    REQUIRE(status.handle(ReadStatus::Overflow, start + 1s) == Action::WarnOverflow);
    REQUIRE(status.deviceOverflows == 102);
    REQUIRE(status.handle(ReadStatus::Overflow, start + 1999ms) == Action::Retry);
    REQUIRE(status.handle(ReadStatus::Overflow, start + 2s) == Action::WarnOverflow);
    REQUIRE(status.deviceOverflows == 104);

    Status otherStream;
    REQUIRE(otherStream.handle(ReadStatus::Overflow, start) == Action::WarnOverflow);
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
                 "Soapy blocks wait for a device selection and disconnect when cleared",
                 "[modules][soapy][block][selection]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    testSoapyState = {};
    const std::string selector = std::string("driver=") + TestSoapyDriver;
    Blocks::Soapy config;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    REQUIRE(config.deviceString.empty());
    REQUIRE(flowgraph->blockCreate("radio", config, {}) == Result::SUCCESS);
    const auto initial = viewBlock("radio");
    REQUIRE(initial.state == Block::State::Incomplete);
    REQUIRE(initial.outputs.empty());
    REQUIRE_FALSE(initial.interfaceOutputs.empty());
    REQUIRE(testSoapyState.lifecycle.empty());
    REQUIRE(testSoapyState.reads->calls == 0);
    const auto device = std::find_if(initial.interfaceConfigs.begin(), initial.interfaceConfigs.end(),
                                    [](const auto& field) { return field.name == "deviceString"; });
    REQUIRE(device != initial.interfaceConfigs.end());
    const auto options = Parser::Get<std::vector<Parser::Map>>(device->format, "options");
    REQUIRE(options.size() >= 2);
    REQUIRE(options.front() == Parser::Map{{"label", "None"}, {"value", ""}});
    const auto option = std::find_if(options.begin(), options.end(), [&](const auto& entry) {
        return Parser::Get<std::string>(entry, "value") == selector;
    });
    REQUIRE(option != options.end());
    REQUIRE(*option == Parser::Map{
        {"label", "CyberEther test device"}, {"value", selector},
    });
    Blocks::Soapy saved;
    REQUIRE(saved.deserialize(initial.config) == Result::SUCCESS);
    REQUIRE(saved.deviceString.empty());
    REQUIRE(flowgraph->blockRecreate("radio", initial.config) == Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Incomplete);
    REQUIRE(testSoapyState.lifecycle.empty());
    REQUIRE(flowgraph->blockReconfigure("radio", {{"frequency", 100.0e6f}}) == Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Incomplete);
    REQUIRE(testSoapyState.lifecycle.empty());

    const Parser::Map selected{{"deviceString", Parser::Get<std::string>(*option, "value")}};
    REQUIRE(flowgraph->blockReconfigure("radio", selected) == Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Created);
    REQUIRE(viewBlock("radio").outputs.contains("signal"));
    REQUIRE(Parser::Get<std::string>(viewBlock("radio").config, "deviceString") == selector);
    REQUIRE(std::count(testSoapyState.lifecycle.begin(), testSoapyState.lifecycle.end(), "make") == 1);
    REQUIRE(flowgraph->blockReconfigure("radio", {{"deviceString", ""}}) == Result::SUCCESS);
    const auto disconnected = viewBlock("radio");
    REQUIRE(disconnected.state == Block::State::Incomplete);
    REQUIRE(disconnected.outputs.empty());
    REQUIRE(saved.deserialize(disconnected.config) == Result::SUCCESS);
    REQUIRE(saved.deviceString.empty());
    REQUIRE(std::count(testSoapyState.lifecycle.begin(), testSoapyState.lifecycle.end(), "make") == 1);
    REQUIRE(std::count(testSoapyState.lifecycle.begin(), testSoapyState.lifecycle.end(), "close") == 1);
    REQUIRE(std::count(testSoapyState.lifecycle.begin(), testSoapyState.lifecycle.end(), "unmake") == 1);
    REQUIRE_FALSE(testSoapyState.releasedWhileReading);
    REQUIRE(flowgraph->blockReconfigure("radio", selected) == Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Created);
    REQUIRE(std::count(testSoapyState.lifecycle.begin(), testSoapyState.lifecycle.end(), "make") == 2);
}

TEST_CASE_METHOD(SoapySelectionFixture,
                 "Soapy blocks retain device selectors independently of display labels",
                 "[modules][soapy][block][selection]") {
    if (Registry::ListAvailableModules("soapy").empty()) {
        SUCCEED("Soapy module is unavailable in this build.");
        return;
    }

    auto first = deviceA;
    auto selected = deviceB;
    first["label"] = "HackRF One #0";
    selected["label"] = "HackRF One #1, test device";
    selected["frontend"] = "required";
    selected["remote"] = "host-a";
    setEntries({first, selected});

    Blocks::Soapy config;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    REQUIRE(flowgraph->blockCreate("radio", config, {}) == Result::SUCCESS);

    const auto deviceOption = [&](const std::string& label) -> Parser::Map {
        const auto block = viewBlock("radio");
        for (const auto& field : block.interfaceConfigs) {
            if (field.name != "deviceString") {
                continue;
            }
            for (const auto& option : Parser::Get<std::vector<Parser::Map>>(field.format, "options")) {
                if (Parser::Get<std::string>(option, "label") == label) {
                    return option;
                }
            }
        }
        FAIL("Device option is missing.");
        return {};
    };
    const auto option = deviceOption(selected.at("label"));
    REQUIRE(option.size() == 2);
    const auto selector = Parser::Get<std::string>(option, "value");
    REQUIRE(SoapySDR::KwargsFromString(selector) == SoapySDR::Kwargs{
        {"driver", TestDiscoveryDriver}, {"serial", "B"},
        {"frontend", "required"}, {"remote", "host-a"},
    });
    REQUIRE(flowgraph->blockReconfigure("radio", {{"deviceString", selector}}) == Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == Block::State::Created);
    REQUIRE(SoapySDR::KwargsFromString(
        Parser::Get<std::string>(deviceOption(first.at("label")), "value")).at("serial") == "A");
    REQUIRE(makeCount() == 1);
    {
        std::lock_guard lock(testDiscoveryState.mutex);
        REQUIRE(testDiscoveryState.lastMakeArgs.at("serial") == "B");
        REQUIRE(testDiscoveryState.lastMakeArgs.at("frontend") == "required");
        REQUIRE(testDiscoveryState.lastMakeArgs.at("remote") == "host-a");
    }

    const auto saved = viewBlock("radio").config;
    REQUIRE(Parser::Get<std::string>(saved, "deviceString") == selector);
    Block::State expected = Block::State::Incomplete;
    std::string expectedLabel = selected.at("label");
    SECTION("label changes and enumeration reordering preserve the selection") {
        selected["label"] = "HackRF One #0";
        first["label"] = "HackRF One #1";
        setEntries({selected, first});
        expected = Block::State::Created;
        expectedLabel = selected.at("label");
    }
    SECTION("custom constructor arguments retain the configured dropdown selection") {
        selected.erase("remote");
        setEntries({first, selected});
        expected = Block::State::Created;
        expectedLabel = "Configured device";
    }
    SECTION("an unavailable selection does not open another device") {
        setEntries({first});
        expectedLabel = "Configured device";
    }
    SECTION("an ambiguous selection does not open the first match") {
        setEntries({selected, selected});
    }
    Modules::SoapyDiscovery::ClearDiscoveryCache();
    REQUIRE(flowgraph->blockRecreate("radio", saved) == Result::SUCCESS);
    REQUIRE(viewBlock("radio").state == expected);
    REQUIRE(Parser::Get<std::string>(viewBlock("radio").config, "deviceString") == selector);
    REQUIRE(Parser::Get<std::string>(deviceOption(expectedLabel), "value") == selector);
    REQUIRE(makeCount() == (expected == Block::State::Created ? 2 : 1));
    if (expected == Block::State::Created) {
        std::lock_guard lock(testDiscoveryState.mutex);
        REQUIRE(testDiscoveryState.lastMakeArgs.at("serial") == "B");
        REQUIRE(testDiscoveryState.lastMakeArgs.at("remote") == "host-a");
    }
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
    config.deviceString = std::string("driver=") + TestSoapyDriver;
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
        REQUIRE(it->format == Parser::Map{{"type", "progressbar"}});
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
