#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <limits>

#include "environment.hh"
#include "jetstream/domains/io/soapy/module.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"
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
        config.sampleRate = 0.0f;
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
        edit = {{"sampleRate", 4.0e6f}};
        failure = "sampleRate";
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
