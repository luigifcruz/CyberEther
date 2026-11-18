#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "domains/io/sdr/devices/soapy/tools/catalog.hh"
#include "domains/io/sdr/devices/soapy/tools/runtime.hh"
#include "domains/io/sdr/devices/sweep.hh"
#include "environment.hh"

using namespace Jetstream;

TEST_CASE("Soapy can load an installed system runtime without opening a radio",
          "[devices][sdr][soapy][runtime][system]") {
    const TestSupport::ScopedEnvironment environment("CYBERETHER_SOAPYSDR_LIBRARY");
    REQUIRE(environment.set(std::nullopt));
    struct Reset {
        Reset() { Radio::SoapyRuntime::Reset(); }
        ~Reset() { Radio::SoapyRuntime::Reset(); }
    } reset;
    const auto library = Radio::SoapyRuntime::Acquire();
    if (!library) {
        SKIP(Radio::SoapyRuntime::Availability().message);
    }
    REQUIRE(Radio::SoapyRuntime::Availability().available);
    REQUIRE(std::string(library->api().getABIVersion()) == Radio::SoapyAbi::Version);
    REQUIRE_FALSE(library->path().empty());
}

TEST_CASE("SoapySDR catalog keeps the device serial when it fits a selector",
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
    CHECK(entry->info.biasTee);
    CHECK(entry->args.at("type") == "b200");
}

TEST_CASE("SoapySDR catalog derives a stable serial when none fits", "[devices][sdr][soapy]") {
    const Radio::SoapyKwargs args{{"driver", "fake"}, {"serial", "a,b=c"}};
    const auto first = Radio::SoapyCatalog::Describe(args);
    const auto second = Radio::SoapyCatalog::Describe(args);
    const auto other = Radio::SoapyCatalog::Describe({{"driver", "fake"}, {"serial", "x,y"}});
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(other);
    CHECK(first->info.serial == "args-cc269433bec6c3e5");
    CHECK(first->info.serial == second->info.serial);
    CHECK(first->info.serial != other->info.serial);
    CHECK(first->info.serial.find_first_of(",=") == std::string::npos);
    CHECK(first->info.label == "fake (SoapySDR)");

    auto labeled = args;
    labeled["label"] = "Second";
    const auto relabeled = Radio::SoapyCatalog::Describe(labeled);
    REQUIRE(relabeled);
    CHECK(relabeled->info.serial != first->info.serial);
}

TEST_CASE("SoapySDR catalog skips devices that are not radios", "[devices][sdr][soapy]") {
    CHECK(Radio::SoapyCatalog::Skipped({{"driver", "audio"}}));
    CHECK_FALSE(Radio::SoapyCatalog::Skipped({{"driver", "uhd"}}));
    CHECK_FALSE(Radio::SoapyCatalog::Skipped({{"driver", "remote"}, {"remote:driver", "audio"}}));
    const auto entries = Radio::SoapyCatalog::List({
        {{"driver", "audio"}, {"label", "Speakers"}},
        {{"driver", "uhd"}, {"serial", "1"}},
    });
    REQUIRE(entries.size() == 1);
    CHECK(entries.front().info.serial == "1");
}

TEST_CASE("SoapySDR catalog picks the highest reported rate under the ceiling",
          "[devices][sdr][soapy]") {
    CHECK(Radio::SoapyCatalog::SweepRate({{1.0e6, 61.44e6}}, 20.0e6) == 20.0e6);
    CHECK(Radio::SoapyCatalog::SweepRate({{32.0e6, 32.0e6}, {16.0e6, 16.0e6}}, 20.0e6) == 16.0e6);
    CHECK(Radio::SoapyCatalog::SweepRate({{8.0e6, 8.0e6}, {1.0e6, 2.0e6}}, 20.0e6) == 8.0e6);
    CHECK(Radio::SoapyCatalog::SweepRate({{64.0e6, 64.0e6}, {32.0e6, 32.0e6}}, 20.0e6) == 32.0e6);
    CHECK(Radio::SoapyCatalog::SweepRate({}, 20.0e6) == 0.0);
}

#if !defined(JST_OS_BROWSER) && !defined(JST_OS_IOS) && !defined(JST_OS_ANDROID)

namespace {

namespace Abi = Radio::SoapyAbi;

struct FakeRadio {
    bool burstSupported = true;
    bool endsBursts = true;
    bool flagsBurstEnd = true;
    F64 rateOffset = 0.0;
    F64 sampleRate = 0.0;
    bool bandwidthSupported = true;
    F64 bandwidth = 0.0;
    U64 zeroReads = 0;
    U64 bursts = 0;
    U64 starts = 0;
    U64 stops = 0;
    bool streaming = false;
    U64 remaining = 0;
    std::vector<std::string> biasTee;
};

FakeRadio& Fake() {
    static FakeRadio radio;
    return radio;
}

char DriverKey[] = "driver";
char DriverValue[] = "fake";
char SerialKey[] = "serial";
char SerialValue[] = "1234";
char BiasTeeKey[] = "biastee";

Radio::SoapyApi FakeApi() {
    Radio::SoapyApi api;
    api.lastError = [] { return ""; };
    api.lastStatus = [] { return 0; };
    api.errToStr = [](int) { return "fake error"; };
    api.free = [](void*) {};
    api.stringsClear = [](char***, std::size_t) {};
    api.kwargsListClear = [](Abi::Kwargs*, std::size_t) {};
    api.argInfoListClear = [](Abi::ArgInfo*, std::size_t) {};
    api.enumerate = [](const Abi::Kwargs*, std::size_t* count) {
        static char* keys[] = {DriverKey, SerialKey};
        static char* values[] = {DriverValue, SerialValue};
        static Abi::Kwargs entry{2, keys, values};
        *count = 1;
        return &entry;
    };
    api.make = [](const Abi::Kwargs*) { return reinterpret_cast<Abi::Device*>(&Fake()); };
    api.unmake = [](Abi::Device*) { return 0; };
    api.getFrequencyRange = [](const Abi::Device*, int, std::size_t, std::size_t* count) {
        static Abi::Range range{10.0e6, 6000.0e6, 0.0};
        *count = 1;
        return &range;
    };
    api.getSampleRateRange = [](const Abi::Device*, int, std::size_t, std::size_t* count) {
        static Abi::Range ranges[2];
        ranges[0] = {16.0e6, 16.0e6, 0.0};
        ranges[1] = {1.0e6, 1.0e6, 0.0};
        *count = 2;
        return ranges;
    };
    api.getSettingInfo = [](const Abi::Device*, std::size_t* count) {
        static Abi::ArgInfo setting{};
        setting.key = BiasTeeKey;
        *count = 1;
        return &setting;
    };
    api.writeSetting = [](Abi::Device*, const char* key, const char* value) {
        if (std::string(key) == "biastee") {
            Fake().biasTee.emplace_back(value);
        }
        return 0;
    };
    api.getNumChannels = [](const Abi::Device*, int) -> std::size_t { return 1; };
    api.listAntennas = [](const Abi::Device*, int, std::size_t, std::size_t* count) -> char** {
        *count = 0;
        return nullptr;
    };
    api.hasGainMode = [](const Abi::Device*, int, std::size_t) { return true; };
    api.setGainMode = [](Abi::Device*, int, std::size_t, bool) { return 0; };
    api.getGainRange = [](const Abi::Device*, int, std::size_t) {
        return Abi::Range{0.0, 60.0, 0.0};
    };
    api.setGain = [](Abi::Device*, int, std::size_t, double) { return 0; };
    api.setSampleRate = [](Abi::Device*, int, std::size_t, double value) {
        Fake().sampleRate = value + Fake().rateOffset;
        return 0;
    };
    api.getSampleRate = [](const Abi::Device*, int, std::size_t) { return Fake().sampleRate; };
    api.setBandwidth = [](Abi::Device*, int, std::size_t, double value) {
        if (!Fake().bandwidthSupported) {
            return -1;
        }
        Fake().bandwidth = value;
        return 0;
    };
    api.setFrequency = [](Abi::Device*, int, std::size_t, double, const Abi::Kwargs*) {
        return 0;
    };
    api.getFrequency = [](const Abi::Device*, int, std::size_t) { return 0.0; };
    api.setupStream = [](Abi::Device*, int, const char*, const std::size_t*, std::size_t,
                         const Abi::Kwargs*) {
        return reinterpret_cast<Abi::Stream*>(&Fake());
    };
    api.closeStream = [](Abi::Device*, Abi::Stream*) { return 0; };
    api.activateStream = [](Abi::Device*, Abi::Stream*, int flags, long long, std::size_t size) {
        auto& fake = Fake();
        if (flags & Abi::EndBurst) {
            if (!fake.burstSupported) {
                return Abi::NotSupported;
            }
            ++fake.bursts;
            fake.remaining = size;
        } else {
            ++fake.starts;
        }
        fake.streaming = true;
        return 0;
    };
    api.deactivateStream = [](Abi::Device*, Abi::Stream*, int, long long) {
        ++Fake().stops;
        Fake().streaming = false;
        return 0;
    };
    api.readStream = [](Abi::Device*, Abi::Stream*, void* const*, std::size_t count, int* flags,
                        long long*, long) {
        auto& fake = Fake();
        *flags = 0;
        if (!fake.streaming) {
            return Abi::Timeout;
        }
        if (fake.zeroReads > 0) {
            --fake.zeroReads;
            return 0;
        }
        const bool burst = fake.remaining > 0 && fake.endsBursts;
        const auto read = burst ? std::min<std::size_t>(count, fake.remaining) : count;
        if (burst) {
            fake.remaining -= read;
            if (fake.remaining == 0) {
                fake.streaming = false;
                *flags = fake.flagsBurstEnd ? Abi::EndBurst : 0;
            }
        }
        return static_cast<int>(read);
    };
    return api;
}

struct FakeRuntime {
    explicit FakeRuntime(const FakeRadio& radio = {}) {
        Fake() = radio;
        Radio::SoapyRuntime::Override(FakeApi());
        Radio::Device::ClearCache();
    }

    ~FakeRuntime() {
        Radio::SoapyRuntime::Reset();
        Radio::Device::ClearCache();
    }
};

std::unique_ptr<Radio::Sweep> OpenSweep(const Radio::Sweep::Settings& settings = {}) {
    std::unique_ptr<Radio::Sweep> sweep;
    REQUIRE(Radio::Sweep::Open("driver=soapy,serial=1234", sweep) == Result::SUCCESS);
    REQUIRE(sweep->configure(settings) == Result::SUCCESS);
    REQUIRE(sweep->start({
        .frequencies = {100.0e6, 200.0e6},
        .sampleRate = settings.sampleRate,
        .captureLength = 1024,
        .dwell = 1,
    }) == Result::SUCCESS);
    return sweep;
}

U64 RunHops(Radio::Sweep& sweep, const U64 hops) {
    U64 captured = 0;
    const auto result = sweep.run([] { return true; }, [&](const Radio::Sweep::Capture& capture) {
        CHECK(capture.samples.size() == 1024);
        return ++captured < hops;
    });
    CHECK(result == Result::SUCCESS);
    return captured;
}

}  // namespace

TEST_CASE("SoapySDR sweep captures one burst per hop", "[devices][sdr][soapy]") {
    const FakeRuntime runtime;
    auto sweep = OpenSweep();

    CHECK(RunHops(*sweep, 3) == 3);
    CHECK(Fake().bursts == 3);
    CHECK(Fake().starts == 0);
}

TEST_CASE("SoapySDR sweep restarts the stream when bursts are not supported",
          "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.burstSupported = false});
    auto sweep = OpenSweep();

    CHECK(RunHops(*sweep, 3) == 3);
    CHECK(Fake().bursts == 0);
    CHECK(Fake().starts == 3);
    CHECK(Fake().stops == 3);
}

TEST_CASE("SoapySDR sweep restarts the stream when bursts never end", "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.endsBursts = false});
    auto sweep = OpenSweep();

    CHECK(RunHops(*sweep, 3) == 3);
    CHECK(Fake().bursts == 1);
    CHECK(Fake().starts == 2);
    CHECK(Fake().stops == 3);
}

TEST_CASE("SoapySDR sweep keeps bursts when the driver ends them silently",
          "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.flagsBurstEnd = false});
    auto sweep = OpenSweep();

    CHECK(RunHops(*sweep, 3) == 3);
    CHECK(Fake().bursts == 3);
    CHECK(Fake().starts == 0);
}

TEST_CASE("SoapySDR sweep treats empty reads as timeouts", "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.zeroReads = 4});
    auto sweep = OpenSweep();

    CHECK(RunHops(*sweep, 2) == 2);
    CHECK(Fake().zeroReads == 0);
}

TEST_CASE("SoapySDR sweep turns the Bias-T off when closed", "[devices][sdr][soapy]") {
    const FakeRuntime runtime;

    SECTION("enabled") {
        auto sweep = OpenSweep({.biasTee = true});
        CHECK(sweep->applied().biasTee);
        sweep.reset();
        CHECK(Fake().biasTee == std::vector<std::string>{"true", "false"});
    }

    SECTION("never enabled") {
        auto sweep = OpenSweep();
        sweep.reset();
        CHECK(Fake().biasTee == std::vector<std::string>{"false"});
    }
}

TEST_CASE("SoapySDR sweep requests rates outside the reported ones", "[devices][sdr][soapy]") {
    SECTION("default rate is a reported rate") {
        const FakeRuntime runtime;
        auto sweep = OpenSweep();
        CHECK(sweep->capabilities().sweepSampleRate == 16.0e6);
    }

    SECTION("rate above the reported ones") {
        const FakeRuntime runtime;
        auto sweep = OpenSweep({.sampleRate = 61.44e6});
        CHECK(Radio::Device::Range::Contains(sweep->capabilities().sampleRateRanges, 61.44e6));
        CHECK(sweep->applied().sampleRate == 61.44e6);
    }

    SECTION("coerced rate") {
        const FakeRuntime runtime({.rateOffset = -1.44e6});
        auto sweep = OpenSweep({.sampleRate = 61.44e6});
        CHECK(sweep->applied().sampleRate == 60.0e6);
    }
}

TEST_CASE("SoapySDR sweep sets the RF bandwidth to the applied rate", "[devices][sdr][soapy]") {
    SECTION("supported") {
        const FakeRuntime runtime({.rateOffset = -1.44e6});
        auto sweep = OpenSweep({.sampleRate = 61.44e6});
        CHECK(Fake().bandwidth == 60.0e6);
    }

    SECTION("rejected") {
        const FakeRuntime runtime({.bandwidthSupported = false});
        auto sweep = OpenSweep();
        CHECK(Fake().bandwidth == 0.0);
        CHECK(sweep->applied().sampleRate == 2.0e6);
    }
}

#endif
