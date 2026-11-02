#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <array>
#include <cstring>
#include <limits>

#include "environment.hh"
#include "jetstream/module_context.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "jetstream/scheduler_context.hh"
#include "jetstream/domains/io/soapy/module.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"
#include "library/soapysdr.hh"
#include "scripted_soapy.hh"
#include "module_impl.hh"

using namespace Jetstream;

namespace {

struct RuntimeReset {
    RuntimeReset() {
        Modules::SoapyRuntime::Reset();
        Modules::SoapyDiscovery::ClearCache();
    }

    ~RuntimeReset() {
        Modules::SoapyRuntime::Reset();
        Modules::SoapyDiscovery::ClearCache();
    }
};

std::shared_ptr<Module> BuildSoapyModule() {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("soapy", DeviceType::CPU, RuntimeType::NATIVE,
                                  "generic", module) == Result::SUCCESS);
    return module;
}

struct SoapyModuleFixture : TestSoapy::Runtime {
    Modules::Soapy config;
    const std::shared_ptr<Module> module = BuildSoapyModule();

    SoapyModuleFixture() {
        config.deviceString = "driver=scripted";
        config.numberOfBatches = 1;
        config.numberOfTimeSamples = 8;
    }

    ~SoapyModuleFixture() { static_cast<void>(module->destroy()); }
};

template<typename Predicate>
bool WaitFor(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return predicate();
}

}  // namespace

TEST_CASE("Soapy validates configuration before accessing the runtime",
          "[modules][soapy][validation]") {
    const TestSupport::ScopedEnvironment environment("CYBERETHER_SOAPYSDR_LIBRARY");
    REQUIRE(environment.set("/__cyberether_missing_soapy__/library"));
    const RuntimeReset reset;
    Modules::Soapy config;
    config.deviceString = "driver=saved,serial=123";
    SECTION("invalid sample rate") {
        config.sampleRate = 0.0f;
    }
    SECTION("non-finite frequency") {
        config.frequency = std::numeric_limits<F32>::quiet_NaN();
    }
    SECTION("non-finite manual gain") {
        config.manualGain = std::numeric_limits<F32>::infinity();
    }
    SECTION("invalid gain mode") {
        config.gainMode = "unknown";
    }
    SECTION("negative bandwidth") {
        config.bandwidth = -1.0;
    }
    SECTION("non-finite stage gain") {
        config.gains = {{"LNA", std::numeric_limits<F64>::quiet_NaN()}};
    }
    SECTION("overflowing buffer dimensions") {
        config.numberOfBatches = std::numeric_limits<U64>::max();
    }
    SECTION("unsupported sample type") {
        config.dataType = "F32";
    }

    const auto module = BuildSoapyModule();
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->outputs().empty());
    REQUIRE(module->outputs().empty());
    REQUIRE(static_cast<const Modules::Soapy&>(module->config()).deviceString.empty());
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE_METHOD(SoapyModuleFixture,
                 "Soapy preserves samples and shape in every supported format",
                 "[modules][soapy][formats]") {
    const auto format = GENERATE(0, 1, 2, 3, 4);
    const auto& selected = Modules::SoapySampleFormats[format];
    CAPTURE(selected.name);
    auto& script = TestSoapy::State();
    const U64 sampleBytes = DataTypeSize(selected.dtype);
    script.samples.resize(8 * sampleBytes);
    for (U64 i = 0; i < script.samples.size(); ++i) {
        script.samples[i] = static_cast<I8>(i % 128);
    }
    script.reads = {3, Modules::SoapyAbi::Timeout, 5};
    config.dataType = std::string(DataTypeToName(selected.dtype));
    config.numberOfBatches = 2;
    config.numberOfTimeSamples = 4;
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    const auto queries = script.queries.load();
    REQUIRE(script.streamFormat == selected.name);
    REQUIRE(WaitFor([&] {
        return module->context()->scheduler()->hasPendingCompute() == Result::SUCCESS;
    }));
    const auto runtime = std::dynamic_pointer_cast<NativeCpuRuntimeContext>(
        module->context()->runtime());
    REQUIRE(runtime);
    REQUIRE(runtime->computeSubmit() == Result::SUCCESS);
    const auto& output = module->outputs().at("signal").tensor;
    REQUIRE(output.dtype() == selected.dtype);
    REQUIRE(output.shape() == Shape{2, 4});
    REQUIRE(std::memcmp(output.data(), script.samples.data(), 8 * sampleBytes) == 0);
    REQUIRE(std::any_cast<F32>(output.attribute("sampleScale")) ==
            1.0f / selected.fullScale);
    REQUIRE(module->context()->scheduler()->hasPendingCompute() == Result::TIMEOUT);
    REQUIRE(runtime->computeSubmit() == Result::YIELD);
    REQUIRE(WaitFor([&] {
        return module->getImpl<Modules::SoapyImpl>()->getThroughput().second ==
               config.sampleRate * sampleBytes / 1e6f;
    }));
    REQUIRE(module->getImpl<Modules::SoapyImpl>()->getDeviceInfo().connected);
    REQUIRE(script.queries == queries);
    REQUIRE(script.concurrentChanges == 0);
}

TEST_CASE_METHOD(TestSoapy::Runtime,
                 "Soapy only offers implemented formats advertised by the device",
                 "[modules][soapy][formats]") {
    TestSoapy::State().streamFormats = {"CS12", "CS16", "CF32", "CS16"};
    Modules::SoapyReceiver receiver;
    REQUIRE(receiver.open("driver=scripted") == Result::SUCCESS);
    REQUIRE(*receiver.info().capabilities.dataTypes.value == std::vector<std::string>{"CF32", "CI16"});
    REQUIRE(receiver.startStream("", DataType::CI8) == Result::ERROR);
    REQUIRE(TestSoapy::State().streamFormat.empty());
    REQUIRE(receiver.startStream("", DataType::CI16) == Result::SUCCESS);
    REQUIRE(TestSoapy::State().streamFormat == "CS16");
}

TEST_CASE_METHOD(TestSoapy::Runtime,
                 "Soapy scaling only uses a valid full scale for the requested format",
                 "[modules][soapy][formats][scale]") {
    auto& script = TestSoapy::State();
    DataType dtype = DataType::CI16;
    F32 expected = 1.0f / 32768.0f;
    SECTION("native signed integers") {
        script.fullScale = 8192.0;
        expected = 1.0f / 8192.0f;
    }
    SECTION("converted integers") {
        script.nativeFormat = "CS8";
        script.fullScale = 128.0;
    }
    SECTION("invalid native range") { script.fullScale = 65536.0; }
    SECTION("zero native range") { script.fullScale = 0.0; }
    SECTION("non-finite native range") {
        script.fullScale = std::numeric_limits<F64>::quiet_NaN();
    }
    SECTION("missing native query") {
        auto api = TestSoapy::Api();
        api.getNativeStreamFormat = nullptr;
        Modules::SoapyRuntime::Override(api);
    }
    SECTION("floating point") {
        dtype = DataType::CF32;
        script.fullScale = 8192.0;
        expected = 1.0f;
    }
    SECTION("unsigned midpoint uses the format width") {
        dtype = DataType::CU8;
        expected = 1.0f / 128.0f;
    }
    Modules::SoapyReceiver receiver;
    REQUIRE(receiver.open("driver=scripted") == Result::SUCCESS);
    REQUIRE(receiver.startStream("", dtype) == Result::SUCCESS);
    REQUIRE(receiver.sampleScale() == expected);
}

TEST_CASE_METHOD(SoapyModuleFixture,
                 "Soapy defaults to CF32 and recreates its stream for format changes",
                 "[modules][soapy][formats][reconfigure]") {
    REQUIRE(config.deserialize({{"deviceString", std::string("driver=scripted")}}) ==
            Result::SUCCESS);
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    REQUIRE(TestSoapy::State().streamFormat == "CF32");
    for (const auto& selected : Modules::SoapySampleFormats) {
        const std::string type(DataTypeToName(selected.dtype));
        const auto result = module->reconfigure({{"dataType", type}});
        if (config.dataType == type) {
            REQUIRE(result == Result::SUCCESS);
        } else {
            REQUIRE(result == Result::RECREATE);
            REQUIRE(module->destroy() == Result::SUCCESS);
            config.dataType = type;
            REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        }
        REQUIRE(module->outputs().at("signal").tensor.dtype() == selected.dtype);
        REQUIRE(TestSoapy::State().streamFormat == selected.name);
    }
    REQUIRE(TestSoapy::State().concurrentChanges == 0);
}

TEST_CASE_METHOD(SoapyModuleFixture,
                 "Soapy discards partial integer frames on overflow",
                 "[modules][soapy][formats][overflow]") {
    auto& script = TestSoapy::State();
    const std::vector<CI16> samples{{1, -1}, {2, -2}, {3, -3}, {4, -4},
                                   {5, -5}, {6, -6}};
    script.samples.resize(samples.size() * sizeof(CI16));
    std::memcpy(script.samples.data(), samples.data(), script.samples.size());
    script.reads = {2, Modules::SoapyAbi::Overflow, 4};
    config.dataType = "CI16";
    config.numberOfTimeSamples = 4;
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    const auto runtime = std::dynamic_pointer_cast<NativeCpuRuntimeContext>(
        module->context()->runtime());
    REQUIRE(runtime);
    REQUIRE(WaitFor([&] { return runtime->computeSubmit() == Result::SUCCESS; }));
    const auto& output = module->outputs().at("signal").tensor;
    REQUIRE(output.data<CI16>()[0] == samples[2]);
    REQUIRE(output.data<CI16>()[3] == samples[5]);
    REQUIRE(module->getImpl<Modules::SoapyImpl>()->getDeviceOverflows() == 1);
}

TEST_CASE_METHOD(TestSoapy::Runtime,
                 "Soapy reads use sample counts and reject oversized driver results",
                 "[modules][soapy][formats][read]") {
    auto api = TestSoapy::Api();
    api.readStream = [](Modules::SoapyAbi::Device*, Modules::SoapyAbi::Stream*,
                         void* const*, std::size_t count, int*, long long*, long) {
        CHECK(count == 8);
        return 9;
    };
    Modules::SoapyRuntime::Override(api);
    Modules::SoapyReceiver receiver;
    std::array<CI16, 8> samples{};
    REQUIRE(receiver.read(samples.data(), samples.size()).status ==
            Modules::SoapyReceiver::ReadStatus::Error);
    REQUIRE(receiver.open("driver=scripted") == Result::SUCCESS);
    REQUIRE(receiver.startStream("", DataType::CI16) == Result::SUCCESS);
    REQUIRE(receiver.read(nullptr, 8).status ==
            Modules::SoapyReceiver::ReadStatus::Error);
    REQUIRE(receiver.read(samples.data(), 0).status ==
            Modules::SoapyReceiver::ReadStatus::Error);
    REQUIRE(receiver.read(samples.data(), samples.size()).error ==
            "Device returned more samples than requested");
}

TEST_CASE("Stream buffers count complete samples for every Soapy format",
          "[modules][soapy][formats][buffer]") {
    const U64 bytes = GENERATE(2, 4, 8, 16);
    Radio::StreamBuffer stream;
    REQUIRE(stream.resize(4, bytes) == Result::SUCCESS);
    REQUIRE(stream.capacity() == 4);
    stream.reset({.generation = 1});
    std::vector<I8> samples(6 * bytes);
    for (U64 i = 0; i < samples.size(); ++i) {
        samples[i] = static_cast<I8>(i);
    }
    std::vector<I8> output(4 * bytes);
    Radio::StreamBuffer::Metadata metadata;
    REQUIRE(stream.push(1, samples.data(), 6));
    REQUIRE(stream.size() == 4);
    REQUIRE(stream.statistics().overwrittenSamples == 2);
    REQUIRE(stream.pop(output.data(), 4, metadata) == Result::SUCCESS);
    REQUIRE(std::memcmp(output.data(), samples.data() + 2 * bytes, output.size()) == 0);
    REQUIRE(stream.push(1, samples.data(), 2, 0));
    REQUIRE(stream.push(1, samples.data(), 2, 4));
    REQUIRE(stream.pop(output.data(), 4, metadata) == Result::INCOMPLETE);
    REQUIRE(stream.statistics().discardedSamples == 2);
    REQUIRE(stream.push(1, samples.data(), 2, 6));
    REQUIRE(stream.pop(output.data(), 4, metadata) == Result::SUCCESS);
    stream.reset({.generation = 2});
    REQUIRE_FALSE(stream.push(1, samples.data(), 2));
    REQUIRE(stream.statistics().rejectedSamples == 2);
    REQUIRE(stream.statistics().pushedSamples == 12);
    REQUIRE(stream.statistics().overwrittenSamples == 2);
    REQUIRE(stream.resize(std::numeric_limits<U64>::max(), bytes) == Result::ERROR);
    REQUIRE(stream.resize(4, 0) == Result::ERROR);
    REQUIRE(stream.capacity() == 4);
}

TEST_CASE("Soapy missing runtime retains saved module configuration",
          "[modules][soapy][runtime]") {
    const TestSupport::ScopedEnvironment environment("CYBERETHER_SOAPYSDR_LIBRARY");
    REQUIRE(environment.set("/__cyberether_missing_soapy__/library"));
    const RuntimeReset reset;
    Modules::Soapy config;
    config.modulePath = "saved-device-module";
    config.deviceString = "driver=saved,serial=123";
    config.streamString = "bufflen=8192";
    Parser::Map saved;
    REQUIRE(config.serialize(saved) == Result::SUCCESS);

    const auto module = BuildSoapyModule();
    REQUIRE(module->create("test", config, {}) == Result::INCOMPLETE);
    REQUIRE(module->state() == Module::State::INCOMPLETE);
    REQUIRE(module->interface()->outputs() == std::vector<std::string>{"signal"});
    REQUIRE(module->outputs().empty());
    Parser::Map retained;
    REQUIRE(module->config(retained) == Result::SUCCESS);
    REQUIRE(retained == saved);
    REQUIRE_FALSE(Modules::SoapyRuntime::Acquire());
    REQUIRE_FALSE(Modules::SoapyRuntime::Availability().message.empty());
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Soapy can load an installed system runtime without opening a radio",
          "[modules][soapy][runtime][system]") {
    const TestSupport::ScopedEnvironment environment("CYBERETHER_SOAPYSDR_LIBRARY");
    REQUIRE(environment.set(std::nullopt));
    const RuntimeReset reset;
    const auto library = Modules::SoapyRuntime::Acquire();
    if (!library) {
        SKIP(Modules::SoapyRuntime::Availability().message);
    }
    REQUIRE(Modules::SoapyRuntime::Availability().available);
    REQUIRE(std::string(library->api().getABIVersion()) == Modules::SoapyAbi::Version);
    REQUIRE_FALSE(library->path().empty());
}

TEST_CASE_METHOD(SoapyModuleFixture,
                 "Soapy gain control respects device capabilities during startup and reconfiguration",
                 "[modules][soapy][devices][agc]") {
    const bool supported = GENERATE(true, false);
    const bool enabled = GENERATE(true, false);
    CAPTURE(supported, enabled);
    TestSoapy::State().automaticGainSupported = supported;

    config.automaticGain = enabled;
    config.manualGain = 30.0f;
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    const auto outputId = module->outputs().at("signal").tensor.id();
    U64 expectedWrites = supported ? 1 : 0;
    REQUIRE(TestSoapy::State().gainModeWrites == expectedWrites);
    REQUIRE(TestSoapy::State().settings().automaticGain == (supported && enabled));

    for (const bool gain : {!enabled, enabled}) {
        REQUIRE(module->reconfigure({{"automaticGain", gain}}) == Result::SUCCESS);
        REQUIRE(module->state() == Module::State::CREATED);
        REQUIRE(module->outputs().at("signal").tensor.id() == outputId);
        REQUIRE(static_cast<const Modules::Soapy&>(module->config()).automaticGain == gain);
        if (supported) {
            ++expectedWrites;
        }
        REQUIRE(TestSoapy::State().gainModeWrites == expectedWrites);
        REQUIRE(TestSoapy::State().settings().automaticGain == (supported && gain));
    }
    REQUIRE(TestSoapy::State().concurrentChanges == 0);
}

TEST_CASE_METHOD(SoapyModuleFixture,
                 "Soapy reports gain control failures on devices advertising AGC",
                 "[modules][soapy][devices][agc]") {
    TestSoapy::State().fail("automaticGain");
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(TestSoapy::State().gainModeWrites == 1);
    REQUIRE(module->state() == Module::State::DESTROYED);
}

TEST_CASE_METHOD(SoapyModuleFixture,
                 "Soapy live edits finish synchronously and failed transactions remain retryable",
                 "[modules][soapy][reconfigure][control][readback]") {
    config.frequency = 100.0e6f;
    config.sampleRate = 2.0e6f;
    config.automaticGain = false;
    config.manualGain = 30.0f;
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);

    Parser::Map edit;
    std::string failure;
    SECTION("frequency") {
        edit = {{"frequency", 300.0e6f}};
        failure = "frequency";
    }
    SECTION("sample rate") {
        edit = {{"sampleRate", 4.0e6f}};
        failure = "sampleRate";
    }
    SECTION("sample rate readback") {
        edit = {{"sampleRate", 4.0e6f}};
        failure = "getSampleRate";
    }
    SECTION("automatic gain") {
        edit = {{"automaticGain", true}};
        failure = "automaticGain";
    }
    SECTION("manual gain after changing the rate and frequency") {
        edit = {{"sampleRate", 4.0e6f}, {"frequency", 300.0e6f}, {"manualGain", 40.0f}};
        failure = "manualGain";
    }
    const auto checkApplied = [&] {
        const auto applied = TestSoapy::State().settings();
        const auto info = module->getImpl<Modules::SoapyImpl>()->getDeviceInfo();
        REQUIRE(applied.frequency == config.frequency);
        REQUIRE(applied.sampleRate == config.sampleRate);
        REQUIRE(applied.automaticGain == config.automaticGain);
        REQUIRE(applied.gain == config.manualGain);
        REQUIRE(info.state.frequency.value == applied.frequency);
        REQUIRE(info.state.sampleRate.value == applied.sampleRate);
        REQUIRE(info.state.automaticGain.value == applied.automaticGain);
        REQUIRE(info.state.gain.value == applied.gain);
    };
    REQUIRE(WaitFor([] { return TestSoapy::State().activeReads > 0; }));
    TestSoapy::State().fail(failure);
    REQUIRE(module->reconfigure(edit) == Result::ERROR);
    checkApplied();
    Parser::Map before;
    Parser::Map retained;
    REQUIRE(config.serialize(before) == Result::SUCCESS);
    REQUIRE(module->config(retained) == Result::SUCCESS);
    REQUIRE(retained == before);

    REQUIRE(module->reconfigure(edit) == Result::SUCCESS);
    REQUIRE(config.deserialize(edit) == Result::SUCCESS);
    checkApplied();
    REQUIRE(TestSoapy::State().concurrentChanges == 0);
}

TEST_CASE_METHOD(SoapyModuleFixture,
                 "Soapy publishes actual tuning while preserving requested settings",
                 "[modules][soapy][readback]") {
    auto& script = TestSoapy::State();
    script.sampleRateStep = 1.0e6;
    script.frequencyStep = 1000.0;
    config.sampleRate = 2.4e6f;
    config.frequency = 100000320.0f;
    config.automaticGain = false;
    config.manualGain = 80.0f;
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    const auto info = module->getImpl<Modules::SoapyImpl>()->getDeviceInfo();
    REQUIRE(info.connected);
    REQUIRE(info.state.sampleRate.value == 2.0e6);
    REQUIRE(info.state.frequency.value == 100.0e6);
    REQUIRE(info.state.gain.value == 60.0);
    const auto& output = module->outputs().at("signal").tensor;
    REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == 2.0e6f);
    REQUIRE(std::any_cast<F32>(output.attribute("frequency")) == 100.0e6f);
    const auto& saved = static_cast<const Modules::Soapy&>(module->config());
    REQUIRE(saved.sampleRate == config.sampleRate);
    REQUIRE(saved.frequency == config.frequency);
    REQUIRE(saved.manualGain == config.manualGain);
}

TEST_CASE_METHOD(SoapyModuleFixture,
                 "Soapy refuses unverified or invalid signal metadata",
                 "[modules][soapy][readback][validation]") {
    auto api = TestSoapy::Api();
    SECTION("missing sample rate getter") {
        api.getSampleRate = nullptr;
    }
    SECTION("non-finite frequency") {
        api.getFrequency = [](const Modules::SoapyAbi::Device*, int, std::size_t) {
            return std::numeric_limits<F64>::quiet_NaN();
        };
    }
    SECTION("zero sample rate") {
        api.getSampleRate = [](const Modules::SoapyAbi::Device*, int, std::size_t) {
            return 0.0;
        };
    }
    SECTION("sample rate cannot fit signal metadata") {
        api.getSampleRate = [](const Modules::SoapyAbi::Device*, int, std::size_t) {
            return std::numeric_limits<F64>::max();
        };
    }
    Modules::SoapyRuntime::Override(api);
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->state() == Module::State::DESTROYED);
    REQUIRE_FALSE(module->getImpl<Modules::SoapyImpl>()->getDeviceInfo().connected);
}

TEST_CASE_METHOD(SoapyModuleFixture,
                 "Soapy isolates missing, failed, and invalid optional queries",
                 "[modules][soapy][capabilities][validation]") {
    auto api = TestSoapy::Api();
    SECTION("gain range query fails") {
        api.getGainRange = [](const Modules::SoapyAbi::Device*, int, std::size_t) {
            TestSoapy::State().query("gainRange");
            TestSoapy::State().queryStatus = -1;
            return Modules::SoapyAbi::Range{};
        };
        api.getGain = nullptr;
        api.getAntenna = nullptr;
    }
    SECTION("gain range is invalid") {
        TestSoapy::State().gainRange = GENERATE(
            Modules::SoapyAbi::Range{60.0, 0.0, 0.0},
            Modules::SoapyAbi::Range{0.0, 60.0, -1.0},
            Modules::SoapyAbi::Range{0.0, std::numeric_limits<F64>::quiet_NaN(), 0.0},
            Modules::SoapyAbi::Range{0.0, std::numeric_limits<F64>::max(), 0.0});
    }
    const bool readbackAvailable = api.getGain != nullptr;
    Modules::SoapyRuntime::Override(api);
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    const auto info = module->getImpl<Modules::SoapyImpl>()->getDeviceInfo();
    REQUIRE_FALSE(info.capabilities.gain.value);
    REQUIRE_FALSE(info.capabilities.gain.error.empty());
    REQUIRE(info.capabilities.biasTee.value == false);
    REQUIRE(info.capabilities.biasTee.error.empty());
    REQUIRE(info.state.gain.value.has_value() == readbackAvailable);
    REQUIRE(info.state.gain.error.empty() == readbackAvailable);
    REQUIRE(info.state.antenna.value.has_value() == readbackAvailable);
    REQUIRE(info.state.sampleRate.value == config.sampleRate);
}

TEST_CASE_METHOD(SoapyModuleFixture,
                 "Soapy lets the driver update rates beyond its initial advertised range",
                 "[modules][soapy][capabilities][readback]") {
    auto api = TestSoapy::Api();
    api.getSampleRateRange = [](const Modules::SoapyAbi::Device*, int, std::size_t,
                                std::size_t* count) {
        auto& script = TestSoapy::State();
        script.query("sampleRateRange");
        script.sampleRateRanges = {{1.0e6, script.settings().sampleRate * 2.0, 0.0}};
        *count = script.sampleRateRanges.size();
        return script.sampleRateRanges.data();
    };
    Modules::SoapyRuntime::Override(api);
    config.sampleRate = 20.0e6f;
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    const auto info = module->getImpl<Modules::SoapyImpl>()->getDeviceInfo();
    REQUIRE(info.state.sampleRate.value == 20.0e6);
    REQUIRE(info.capabilities.sampleRates.value);
    REQUIRE(info.capabilities.sampleRates.value->front().maximum == 40.0e6);
}

TEST_CASE_METHOD(SoapyModuleFixture,
                 "Soapy refreshes gain limits without losing the requested gain",
                 "[modules][soapy][capabilities][reconfigure]") {
    auto api = TestSoapy::Api();
    api.getGainRange = [](const Modules::SoapyAbi::Device*, int, std::size_t) {
        auto& script = TestSoapy::State();
        script.query("gainRange");
        const F64 maximum = script.settings().frequency < 200.0e6 ? 60.0 : 20.0;
        return Modules::SoapyAbi::Range{0.0, maximum, 0.0};
    };
    Modules::SoapyRuntime::Override(api);
    config.frequency = 100.0e6f;
    config.automaticGain = false;
    config.manualGain = 50.0f;
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    REQUIRE(module->reconfigure({{"frequency", 300.0e6f}}) == Result::SUCCESS);
    auto info = module->getImpl<Modules::SoapyImpl>()->getDeviceInfo();
    REQUIRE(info.capabilities.gain.value->maximum == 20.0);
    REQUIRE(info.state.gain.value == 20.0);
    REQUIRE(static_cast<const Modules::Soapy&>(module->config()).manualGain == 50.0f);
    REQUIRE(module->reconfigure({{"frequency", 100.0e6f}}) == Result::SUCCESS);
    info = module->getImpl<Modules::SoapyImpl>()->getDeviceInfo();
    REQUIRE(info.capabilities.gain.value->maximum == 60.0);
    REQUIRE(info.state.gain.value == 50.0);
}

TEST_CASE_METHOD(TestSoapy::Runtime,
                 "Soapy still turns off Bias-T after an optional capability query fails",
                 "[modules][soapy][capabilities][cleanup]") {
    auto& script = TestSoapy::State();
    script.biasTeeSupported = true;
    Modules::SoapyReceiver receiver;
    REQUIRE(receiver.open("driver=scripted") == Result::SUCCESS);
    REQUIRE(receiver.setBiasTee(true) == Result::SUCCESS);
    REQUIRE(script.biasTeeEnabled);
    script.fail("settings");
    REQUIRE(receiver.refresh() == Result::SUCCESS);
    REQUIRE_FALSE(receiver.info().capabilities.biasTee.value);
    REQUIRE_FALSE(receiver.info().capabilities.biasTee.error.empty());
    receiver.reset();
    REQUIRE_FALSE(script.biasTeeEnabled);
}

TEST_CASE_METHOD(SoapyModuleFixture,
                 "Soapy stage gains and bandwidth preserve requests and roll back together",
                 "[modules][soapy][controls]") {
    auto& script = TestSoapy::State();
    script.controlsSupported = true;
    config.automaticGain = false;
    config.gainMode = "stages";
    config.gains = {{"LNA", 8.0}, {"VGA", 12.5}};
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    REQUIRE(script.totalGainWrites == 0);
    REQUIRE(script.bandwidthWrites == 0);
    const auto impl = module->getImpl<Modules::SoapyImpl>();
    REQUIRE(impl->getDeviceInfo().state.gains.at("VGA").value == 12.5);
    REQUIRE(module->reconfigure({{"gains", Parser::Map{{"LNA", 10.0}, {"VGA", 14.0}}},
                                 {"bandwidth", 6.4e6}}) == Result::SUCCESS);
    REQUIRE(impl->getDeviceInfo().state.bandwidth.value == 6.0e6);
    REQUIRE(static_cast<const Modules::Soapy&>(module->config()).bandwidth == 6.4e6);
    script.fail("gainElement");
    REQUIRE(module->reconfigure({{"frequency", 110.0e6f}}) == Result::SUCCESS);
    REQUIRE_FALSE(impl->getDeviceInfo().state.gains.at("LNA").value);
    REQUIRE(module->reconfigure({{"frequency", 120.0e6f}}) == Result::SUCCESS);
    const Parser::Map edit{{"gains", Parser::Map{{"LNA", 20.0}, {"VGA", 22.0}}},
                           {"bandwidth", 12.0e6}};
    script.fail("gain:VGA");
    REQUIRE(module->reconfigure(edit) == Result::ERROR);
    REQUIRE(script.gainStages == std::map<std::string, F64>{{"LNA", 10.0}, {"VGA", 14.0}});
    REQUIRE(impl->getDeviceInfo().state.bandwidth.value == 6.0e6);
    REQUIRE(static_cast<const Modules::Soapy&>(module->config()).bandwidth == 6.4e6);
    REQUIRE(module->reconfigure(edit) == Result::SUCCESS);
    REQUIRE(impl->getDeviceInfo().state.bandwidth.value == 12.0e6);
    REQUIRE(module->reconfigure({{"gainMode", "total"}, {"manualGain", 17.0f}}) ==
            Result::SUCCESS);
    REQUIRE(script.totalGainWrites == 1);
    REQUIRE(impl->getDeviceInfo().state.gain.value == 17.0);
    REQUIRE(module->reconfigure({{"bandwidth", 0.0}}) == Result::RECREATE);
    REQUIRE(script.concurrentChanges == 0);
}

TEST_CASE_METHOD(SoapyModuleFixture,
                 "Soapy validates typed driver parameters and applies them before streaming",
                 "[modules][soapy][controls][arguments]") {
    auto& script = TestSoapy::State();
    script.controlsSupported = true;
    script.applied.sampleRate = 0.0;
    config.settings = {{"enabled", true}, {"decimation", I64{4}}, {"correction", 0.75},
                       {"mode", "quiet"}, {"label", "a,b=quoted: value"}};
    config.streamArgs = {{"wire", "sc8"}};
    SECTION("valid settings survive restart requests without live writes") {
        REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        REQUIRE(script.driverSettings.at("enabled") == "true");
        REQUIRE(script.driverSettings.at("decimation") == "4");
        REQUIRE(script.driverSettings.at("correction") == "0.75");
        REQUIRE(script.driverSettings.at("mode") == "quiet");
        REQUIRE(script.driverSettings.at("label") == "a,b=quoted: value");
        REQUIRE(script.streamArgs.at("wire") == "sc8");
        const auto info = module->getImpl<Modules::SoapyImpl>()->getDeviceInfo();
        REQUIRE(info.state.settings.at("mode").value == "quiet");
        REQUIRE(module->reconfigure({{"settings", Parser::Map{{"mode", "fast"}}}}) ==
                Result::RECREATE);
        REQUIRE(module->reconfigure({{"streamArgs", Parser::Map{{"wire", "sc16"}}}}) ==
                Result::RECREATE);
        REQUIRE(script.driverSettings.at("mode") == "quiet");
        REQUIRE(script.concurrentChanges == 0);
    }
    SECTION("invalid settings fail before any writes") {
        const auto invalid = GENERATE(
            std::pair{"enabled", "maybe"}, std::pair{"decimation", "1.5"},
            std::pair{"correction", "nan"}, std::pair{"correction", "2"},
            std::pair{"mode", "unknown"}, std::pair{"unknown", "x"},
            std::pair{"biastee", "true"});
        CAPTURE(invalid.first, invalid.second);
        config.settings = {{invalid.first, invalid.second}};
        const auto before = script.driverSettings;
        REQUIRE(module->create("test", config, {}) == Result::ERROR);
        REQUIRE(script.driverSettings == before);
        REQUIRE_FALSE(script.streamStarted);
    }
    SECTION("invalid stream option") {
        config.streamArgs = {{"wire", "unknown"}};
        REQUIRE(module->create("test", config, {}) == Result::ERROR);
        REQUIRE_FALSE(script.streamStarted);
    }
    SECTION("driver refuses a setting") {
        script.fail("setting:mode");
        REQUIRE(module->create("test", config, {}) == Result::ERROR);
        REQUIRE_FALSE(script.streamStarted);
    }
}
