#include <catch2/catch_test_macros.hpp>

#include <any>
#include <limits>
#include <string>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/adsb/module.hh"

#include "module_impl.hh"

using namespace Jetstream;

namespace {

struct AdsbImplAccess : Modules::AdsbImpl {
    static auto inputMember() {
        return &AdsbImplAccess::input;
    }

    static auto aircraftMember() {
        return &AdsbImplAccess::aircraft;
    }

    static auto aircraftCountMember() {
        return &AdsbImplAccess::aircraftCount;
    }
};

TensorMap AdsbInput(const Tensor& input) {
    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;
    return inputs;
}

void RequireAdsbValidationError(const Registry::ModuleRegistration& impl,
                                const Tensor& input) {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("adsb", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);

    Result result = Result::SUCCESS;
    REQUIRE_NOTHROW(result = module->create("test", Modules::Adsb{},
                                            AdsbInput(input)));
    REQUIRE(result == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());

    const auto* adsb = module->getImpl<Modules::AdsbImpl>();
    REQUIRE(adsb != nullptr);
    REQUIRE((adsb->*AdsbImplAccess::inputMember()).empty());
    REQUIRE((adsb->*AdsbImplAccess::aircraftMember()).empty());
    REQUIRE((adsb->*AdsbImplAccess::aircraftCountMember()).empty());
}

}  // namespace

TEST_CASE("ADS-B - Silence Input",
          "[modules][adsb][silence]") {
    auto implementations = Registry::ListAvailableModules("adsb");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("adsb", impl.device, impl.runtime,
                           impl.provider);

            Modules::Adsb config;
            ctx.setConfig(config);

            const U64 bufferSize = 240;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                  {bufferSize}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("frequency", F32{1090e6f}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleRate", F32{2e6f}) ==
                    Result::SUCCESS);

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

        }
    }
}

TEST_CASE("ADS-B - Random Noise Input",
          "[modules][adsb][noise]") {
    auto implementations = Registry::ListAvailableModules("adsb");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("adsb", impl.device, impl.runtime,
                           impl.provider);

            Modules::Adsb config;
            ctx.setConfig(config);

            // Create low-level noise input.
            const U64 bufferSize = 65536;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                 {bufferSize}) == Result::SUCCESS);

            for (U64 i = 0; i < bufferSize; ++i) {
                const F32 r = static_cast<F32>(i % 7) * 0.001f - 0.003f;
                const F32 q = static_cast<F32>(i % 11) * 0.001f - 0.005f;
                input.at<CF32>(i) = CF32(r, q);
            }

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

        }
    }
}

TEST_CASE("ADS-B - Invalid Input DType",
          "[modules][adsb][validation]") {
    auto implementations = Registry::ListAvailableModules("adsb");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            Tensor input;
            REQUIRE(input.create(impl.device, DataType::F32,
                                  {8192}) == Result::SUCCESS);
            RequireAdsbValidationError(impl, input);
        }
    }
}

TEST_CASE("ADS-B - Malformed Metadata",
          "[modules][adsb][validation][metadata]") {
    const auto implementations = Registry::ListAvailableModules("adsb");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            for (const std::string& key : {std::string("frequency"),
                                           std::string("sampleRate")}) {
                Tensor input;
                REQUIRE(input.create(impl.device, DataType::CF32,
                                     {8192}) == Result::SUCCESS);
                const F64 value = key == "frequency" ? 1090e6 : 2e6;
                REQUIRE(input.setAttribute(key, value) == Result::SUCCESS);
                RequireAdsbValidationError(impl, input);
            }
        }
    }
}

TEST_CASE("ADS-B - Input Size Boundaries",
          "[modules][adsb][validation][size]") {
    const auto implementations = Registry::ListAvailableModules("adsb");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            SECTION("libmodes minimum") {
                Tensor input;
                REQUIRE(input.create(reinterpret_cast<void*>(0x1000),
                                     impl.device,
                                     DataType::CF32,
                                     {239}) == Result::SUCCESS);
                REQUIRE(input.contiguous());
                RequireAdsbValidationError(impl, input);
            }

            SECTION("libmodes sample-count representation") {
                constexpr U64 inputSize =
                    static_cast<U64>(std::numeric_limits<U32>::max()) + 1;
                Tensor input;
                REQUIRE(input.create(reinterpret_cast<void*>(0x1000),
                                     impl.device,
                                     DataType::CF32,
                                     {inputSize}) == Result::SUCCESS);
                REQUIRE(input.contiguous());
                REQUIRE(input.size() == inputSize);
                RequireAdsbValidationError(impl, input);
            }
        }
    }
}

TEST_CASE("ADS-B - Metadata Validation Preserves Live State",
          "[modules][adsb][validation][metadata][rollback]") {
    const auto implementations = Registry::ListAvailableModules("adsb");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            Tensor input;
            REQUIRE(input.create(impl.device, DataType::CF32, {8192}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("frequency", F32{1090e6f}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleRate", F32{2e6f}) ==
                    Result::SUCCESS);

            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("adsb", impl.device, impl.runtime,
                                          impl.provider, module) == Result::SUCCESS);
            REQUIRE(module->create("test", Modules::Adsb{}, AdsbInput(input)) ==
                    Result::SUCCESS);

            auto* adsb = module->getImpl<Modules::AdsbImpl>();
            REQUIRE(adsb != nullptr);
            const Tensor& activeInput = adsb->*AdsbImplAccess::inputMember();
            const Tensor& aircraft = adsb->*AdsbImplAccess::aircraftMember();
            const Tensor& aircraftCount =
                adsb->*AdsbImplAccess::aircraftCountMember();
            const Index aircraftId = aircraft.id();
            const Index aircraftCountId = aircraftCount.id();

            REQUIRE(input.setAttribute("sampleRate", F64{2e6}) ==
                    Result::SUCCESS);
            REQUIRE(adsb->validate() == Result::ERROR);
            REQUIRE(module->state() == Module::State::CREATED);
            REQUIRE(aircraft.id() == aircraftId);
            REQUIRE(aircraftCount.id() == aircraftCountId);

            REQUIRE(input.setAttribute("sampleRate", F32{4e6f}) == Result::SUCCESS);
            REQUIRE(adsb->validate() == Result::ERROR);
            REQUIRE(std::any_cast<F32>(activeInput.attribute("sampleRate")) ==
                    4e6f);

            REQUIRE(input.setAttribute("sampleRate", F32{2e6f}) == Result::SUCCESS);
            REQUIRE(adsb->validate() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
        }
    }
}
