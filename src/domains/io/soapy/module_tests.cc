#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <any>
#include <chrono>
#include <limits>
#include <optional>
#include <string>
#include <thread>

#include "environment.hh"
#include "jetstream/domains/io/soapy/module.hh"
#include "jetstream/memory/attributes.hh"
#include "jetstream/module_context.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime_context_native_cpu.hh"
#include "library/soapysdr.hh"
#include "scripted_soapy.hh"

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

}  // namespace

TEST_CASE("Soapy validates configuration before accessing the runtime",
          "[modules][soapy][validation]") {
    const TestSupport::ScopedEnvironment environment("CYBERETHER_SOAPYSDR_LIBRARY");
    REQUIRE(environment.set("/__cyberether_missing_soapy__/library"));
    const RuntimeReset reset;
    Modules::Soapy config;
    config.deviceString = "driver=saved,serial=123";
    SECTION("invalid sample rate") {
        config.sampleRate = GENERATE(0.0f, -1.0f,
                                    std::numeric_limits<F32>::quiet_NaN(),
                                    std::numeric_limits<F32>::infinity(),
                                    -std::numeric_limits<F32>::infinity());
    }
    SECTION("non-finite frequency") {
        config.frequency = std::numeric_limits<F32>::quiet_NaN();
    }
    SECTION("non-finite manual gain") {
        config.manualGain = std::numeric_limits<F32>::infinity();
    }
    SECTION("overflowing buffer dimensions") {
        config.numberOfBatches = std::numeric_limits<U64>::max();
    }

    CAPTURE(config.sampleRate);
    const auto module = BuildSoapyModule();
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->outputs().empty());
    REQUIRE(module->outputs().empty());
    REQUIRE(static_cast<const Modules::Soapy&>(module->config()).deviceString.empty());
    REQUIRE(module->destroy() == Result::SUCCESS);
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

TEST_CASE_METHOD(TestSoapy::Runtime,
                 "Soapy gain control respects device capabilities during startup and reconfiguration",
                 "[modules][soapy][devices][agc]") {
    const bool supported = GENERATE(true, false);
    const bool enabled = GENERATE(true, false);
    CAPTURE(supported, enabled);
    TestSoapy::State().automaticGainSupported = supported;

    const auto module = BuildSoapyModule();
    struct Cleanup {
        std::shared_ptr<Module> module;
        ~Cleanup() { static_cast<void>(module->destroy()); }
    } cleanup{module};
    Modules::Soapy config;
    config.deviceString = "driver=scripted";
    config.automaticGain = enabled;
    config.manualGain = 30.0f;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
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

TEST_CASE_METHOD(TestSoapy::Runtime,
                 "Soapy reports gain control failures on devices advertising AGC",
                 "[modules][soapy][devices][agc]") {
    TestSoapy::State().fail("automaticGain");
    Modules::Soapy config;
    config.deviceString = "driver=scripted";
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;

    const auto module = BuildSoapyModule();
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(TestSoapy::State().gainModeWrites == 1);
    REQUIRE(module->state() == Module::State::DESTROYED);
}

TEST_CASE_METHOD(TestSoapy::Runtime,
                 "Soapy live edits finish synchronously and failed transactions remain retryable",
                 "[modules][soapy][reconfigure][control]") {
    const auto module = BuildSoapyModule();
    struct Cleanup {
        std::shared_ptr<Module> module;
        ~Cleanup() { static_cast<void>(module->destroy()); }
    } cleanup{module};
    Modules::Soapy config;
    config.deviceString = "driver=scripted";
    config.frequency = 100.0e6f;
    config.sampleRate = 2.0e6f;
    config.automaticGain = false;
    config.manualGain = 30.0f;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);

    Parser::Map edit;
    std::string failure;
    SECTION("frequency") {
        edit = {{"frequency", 300.0e6f}};
        failure = "frequency";
    }
    SECTION("sample rate") {
        edit = {{"sampleRate", 61.44e6f}};
        failure = "sampleRate";
    }
    SECTION("sample rate readback") {
        edit = {{"sampleRate", 61.44e6f}};
        failure = "sampleRateReadback";
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
        REQUIRE(applied.frequency == config.frequency);
        REQUIRE(applied.sampleRate == config.sampleRate);
        REQUIRE(applied.automaticGain == config.automaticGain);
        REQUIRE(applied.gain == config.manualGain);
    };
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (TestSoapy::State().activeReads == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    REQUIRE(TestSoapy::State().activeReads > 0);
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

TEST_CASE_METHOD(TestSoapy::Runtime,
                 "Soapy module forwards positive sample rates to the driver",
                 "[modules][soapy][reconfigure][rate]") {
    const auto module = BuildSoapyModule();
    struct Cleanup {
        std::shared_ptr<Module> module;
        ~Cleanup() { static_cast<void>(module->destroy()); }
    } cleanup{module};
    Modules::Soapy config;
    config.deviceString = "driver=scripted";
    config.sampleRate = 20.0e6f;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    REQUIRE(TestSoapy::State().settings().sampleRate == 20.0e6);
    REQUIRE(module->state() == Module::State::CREATED);
    REQUIRE(std::any_cast<F32>(module->outputs().at("signal").tensor.attribute("sampleRate")) ==
            config.sampleRate);

    REQUIRE(module->reconfigure({{"sampleRate", 61.44e6f}}) == Result::SUCCESS);
    REQUIRE(TestSoapy::State().settings().sampleRate == static_cast<F64>(61.44e6f));
    REQUIRE(module->state() == Module::State::CREATED);
    REQUIRE(static_cast<const Modules::Soapy&>(module->config()).sampleRate == 61.44e6f);
    REQUIRE(TestSoapy::State().concurrentChanges == 0);
}

TEST_CASE_METHOD(TestSoapy::Runtime,
                 "Soapy module publishes the applied sample rate and retains the request",
                 "[modules][soapy][reconfigure][metadata]") {
    TestSoapy::State().sampleRateOffset = 125.0;
    TestSoapy::State().emitSamples = true;
    const auto module = BuildSoapyModule();
    struct Cleanup {
        std::shared_ptr<Module> module;
        ~Cleanup() { static_cast<void>(module->destroy()); }
    } cleanup{module};
    Modules::Soapy config;
    config.deviceString = "driver=scripted";
    config.sampleRate = 20.0e6f;
    config.automaticGain = false;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    const auto& output = module->outputs().at("signal").tensor;
    const auto runtime = std::dynamic_pointer_cast<NativeCpuRuntimeContext>(
        module->context()->runtime());
    REQUIRE(runtime != nullptr);
    const auto checkApplied = [&] {
        REQUIRE(module->state() == Module::State::CREATED);
        REQUIRE(static_cast<const Modules::Soapy&>(module->config()).sampleRate ==
                config.sampleRate);
        const F64 applied = config.sampleRate + TestSoapy::State().sampleRateOffset;
        REQUIRE(TestSoapy::State().settings().sampleRate == applied);
        REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) ==
                static_cast<F32>(applied));
    };
    const auto frame = [&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        Result result;
        do {
            result = runtime->computeSubmit();
            if (result != Result::YIELD) {
                break;
            }
            std::this_thread::yield();
        } while (std::chrono::steady_clock::now() < deadline);
        REQUIRE(result == Result::SUCCESS);
    };
    checkApplied();

    REQUIRE(module->reconfigure({{"sampleRate", 61.44e6f}}) == Result::SUCCESS);
    config.sampleRate = 61.44e6f;
    frame();
    checkApplied();
    REQUIRE(TestSoapy::State().sampleRateWrites == 2);

    REQUIRE(module->reconfigure({{"frequency", 300.0e6f}}) == Result::SUCCESS);
    frame();
    checkApplied();
    REQUIRE(std::any_cast<F32>(output.attribute("frequency")) == 300.0e6f);
    REQUIRE(TestSoapy::State().sampleRateWrites == 2);

    TestSoapy::State().fail("manualGain");
    const Parser::Map edit{{"sampleRate", 30.72e6f}, {"manualGain", 40.0f}};
    REQUIRE(module->reconfigure(edit) == Result::ERROR);
    frame();
    checkApplied();

    REQUIRE(module->reconfigure(edit) == Result::SUCCESS);
    config.sampleRate = 30.72e6f;
    frame();
    checkApplied();
    REQUIRE(TestSoapy::State().concurrentChanges == 0);
}

TEST_CASE_METHOD(TestSoapy::Runtime,
                 "Soapy module rejects invalid sample rate readback without losing settings",
                 "[modules][soapy][validation][reconfigure]") {
    const F64 invalid = GENERATE(0.0, -1.0,
                                 std::numeric_limits<F64>::quiet_NaN(),
                                 std::numeric_limits<F64>::infinity(),
                                 -std::numeric_limits<F64>::infinity(),
                                 std::numeric_limits<F64>::max(),
                                 std::numeric_limits<F64>::min());
    CAPTURE(invalid);
    const auto module = BuildSoapyModule();
    struct Cleanup {
        std::shared_ptr<Module> module;

        ~Cleanup() {
            if (module->state() == Module::State::CREATED) {
                static_cast<void>(module->destroy());
            }
        }
    } cleanup{module};
    Modules::Soapy config;
    config.deviceString = "driver=scripted";
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;

    SECTION("creation") {
        TestSoapy::State().nextSampleRateReadback = invalid;
        REQUIRE(module->create("test", config, {}) == Result::ERROR);
        REQUIRE(module->state() == Module::State::DESTROYED);
    }
    SECTION("reconfiguration") {
        REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
        TestSoapy::State().nextSampleRateReadback = invalid;
        const Parser::Map edit{{"sampleRate", 61.44e6f}};
        REQUIRE(module->reconfigure(edit) == Result::ERROR);
        REQUIRE(module->state() == Module::State::CREATED);
        REQUIRE(TestSoapy::State().settings().sampleRate == config.sampleRate);
        REQUIRE(static_cast<const Modules::Soapy&>(module->config()).sampleRate ==
                config.sampleRate);

        REQUIRE(module->reconfigure(edit) == Result::SUCCESS);
        REQUIRE(TestSoapy::State().settings().sampleRate == static_cast<F64>(61.44e6f));
        REQUIRE(static_cast<const Modules::Soapy&>(module->config()).sampleRate == 61.44e6f);
    }
    REQUIRE(TestSoapy::State().concurrentChanges == 0);
}

TEST_CASE_METHOD(TestSoapy::Runtime,
                 "Soapy module publishes clipped samples and overload signals",
                 "[modules][soapy][overload]") {
    // The scripted device streams 1+0j, which is full scale.
    TestSoapy::State().emitSamples = true;
    // Overload signals read set at start-up must be ignored until clear.
    TestSoapy::State().overloadBits = 0x7;
    const auto module = BuildSoapyModule();
    struct Cleanup {
        std::shared_ptr<Module> module;
        ~Cleanup() { static_cast<void>(module->destroy()); }
    } cleanup{module};
    Modules::Soapy config;
    config.deviceString = "driver=scripted";
    config.sampleRate = 20.0e6f;
    config.numberOfBatches = 1;
    config.numberOfTimeSamples = 8;
    REQUIRE(module->create("test", config, {}) == Result::SUCCESS);
    const auto& output = module->outputs().at("signal").tensor;
    const auto runtime = std::dynamic_pointer_cast<NativeCpuRuntimeContext>(
        module->context()->runtime());
    REQUIRE(runtime != nullptr);

    const auto read = [&](const std::string& key) {
        std::optional<U64> value;
        REQUIRE(ReadAttribute(output, key, value) == Result::SUCCESS);
        return value;
    };
    // Run frames until the attribute matches, or give up after two seconds.
    const auto waitFor = [&](const std::string& key, const auto& accept) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (std::chrono::steady_clock::now() < deadline) {
            const auto result = runtime->computeSubmit();
            REQUIRE((result == Result::SUCCESS || result == Result::YIELD));
            if (const auto value = read(key); value && accept(*value)) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return false;
    };

    REQUIRE(waitFor("clippedSamples", [](const U64 value) { return value > 0; }));
    // Stale start-up signals stay hidden.
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    REQUIRE(waitFor("overloadFlags", [](const U64 value) { return value == 0; }));

    TestSoapy::State().overloadBits = 0;
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    TestSoapy::State().overloadBits = 0x4;
    REQUIRE(waitFor("overloadFlags", [](const U64 value) { return value == 0x4; }));
}
