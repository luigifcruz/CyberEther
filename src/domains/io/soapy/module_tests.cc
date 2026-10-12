#include <catch2/catch_test_macros.hpp>

#include <limits>

#include "environment.hh"
#include "jetstream/domains/io/soapy/module.hh"
#include "jetstream/module_interface.hh"
#include "jetstream/registry.hh"
#include "library/soapysdr.hh"

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
