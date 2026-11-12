#include <catch2/catch_test_macros.hpp>

#include "domains/io/sdr/devices/soapy/tools/catalog.hh"

using namespace Jetstream;

TEST_CASE("Soapy catalog keeps the device serial when it fits a selector",
          "[devices][sdr][soapy]") {
    const auto entry = Radio::SoapyCatalog::Describe({
        {"driver", "uhd"},
        {"label", "B210 [3141592]"},
        {"serial", "3141592"},
        {"type", "b200"},
    });
    REQUIRE(entry);
    CHECK(entry->info.serial == "3141592");
    CHECK(entry->info.label == "B210 [3141592] (SoapySDR)");
    CHECK(entry->args.at("type") == "b200");
}

TEST_CASE("Soapy catalog derives a stable serial when none fits", "[devices][sdr][soapy]") {
    const Modules::SoapyKwargs args{{"driver", "fake"}, {"serial", "a,b=c"}};
    const auto first = Radio::SoapyCatalog::Describe(args);
    const auto second = Radio::SoapyCatalog::Describe(args);
    const auto other = Radio::SoapyCatalog::Describe({{"driver", "fake"}, {"serial", "x,y"}});
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(other);
    CHECK(first->info.serial.starts_with("args-"));
    CHECK(first->info.serial == second->info.serial);
    CHECK(first->info.serial != other->info.serial);
    CHECK(first->info.serial.find_first_of(",=") == std::string::npos);
    CHECK(first->info.label == "fake (SoapySDR)");
}

TEST_CASE("Soapy catalog skips devices that are not radios", "[devices][sdr][soapy]") {
    CHECK(Radio::SoapyCatalog::Native({{"driver", "audio"}}));
    CHECK_FALSE(Radio::SoapyCatalog::Native({{"driver", "uhd"}}));
    const auto entries = Radio::SoapyCatalog::List({
        {{"driver", "audio"}, {"label", "Speakers"}},
        {{"driver", "uhd"}, {"serial", "1"}},
    });
    REQUIRE(entries.size() == 1);
    CHECK(entries.front().info.serial == "1");
}

TEST_CASE("Soapy catalog widens reported rates to the probed rate", "[devices][sdr][soapy]") {
    const std::vector<Radio::Device::Range> reported{
        {16.0e6, 16.0e6}, {8.0e6, 8.0e6}, {31.25e3, 31.25e3}};

    const auto plain = Radio::SoapyCatalog::SampleRates(reported, 0.0);
    REQUIRE(plain.size() == 1);
    CHECK(plain.front().minimum == 31.25e3);
    CHECK(plain.front().maximum == 16.0e6);

    const auto probed = Radio::SoapyCatalog::SampleRates(reported, 61.44e6);
    REQUIRE(probed.size() == 1);
    CHECK(probed.front().minimum == 31.25e3);
    CHECK(probed.front().maximum == 61.44e6);
    CHECK(Radio::Device::Range::Contains(probed, 61.44e6));

    CHECK(Radio::SoapyCatalog::SampleRates({}, 0.0).empty());
}
