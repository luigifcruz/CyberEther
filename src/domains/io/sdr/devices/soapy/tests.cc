#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "domains/io/sdr/devices/soapy/tools/catalog.hh"
#include "domains/io/sdr/devices/soapy/tools/library.hh"
#include "domains/io/sdr/devices/stream.hh"
#include "domains/io/sdr/devices/sweep.hh"
#include "environment.hh"

using namespace Jetstream;

TEST_CASE("SoapySDR can load an installed system runtime without opening a radio",
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
    const std::string version = library->api().getABIVersion();
    REQUIRE(std::find(Radio::SoapyAbi::Versions.begin(), Radio::SoapyAbi::Versions.end(),
                      version) != Radio::SoapyAbi::Versions.end());
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
    CHECK_FALSE(Radio::SoapyCatalog::Skipped({{"driver", "plutosdr"}, {"uri", "ip:192.168.2.1"}}));
    CHECK_FALSE(Radio::SoapyCatalog::Skipped({{"driver", "plutosdr"}, {"uri", "local:"}}));
    CHECK_FALSE(Radio::SoapyCatalog::Skipped({{"driver", "lime"}, {"media", "PCI-E"}}));
    CHECK_FALSE(Radio::SoapyCatalog::Skipped({{"driver", "remote"}, {"remote:driver", "audio"}}));
    const auto entries = Radio::SoapyCatalog::List({
        {{"driver", "audio"}, {"label", "Speakers"}},
        {{"driver", "uhd"}, {"serial", "1"}},
    });
    REQUIRE(entries.size() == 1);
    CHECK(entries.front().info.serial == "1");
}

TEST_CASE("SoapySDR catalog tells apart devices that share a serial", "[devices][sdr][soapy]") {
    const auto entries = Radio::SoapyCatalog::List({
        {{"driver", "usdr"}, {"serial", "0001"}, {"bus", "pci:0"}},
        {{"driver", "usdr"}, {"serial", "0001"}, {"bus", "pci:1"}},
        {{"driver", "uhd"}, {"serial", "3141592"}},
    });
    REQUIRE(entries.size() == 3);
    CHECK(entries[0].info.serial.starts_with("args-"));
    CHECK(entries[1].info.serial.starts_with("args-"));
    CHECK(entries[0].info.serial != entries[1].info.serial);
    CHECK(entries[2].info.serial == "3141592");
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
    F64 rateOffset = 0.0;
    F64 sampleRate = 0.0;
    bool bandwidthSupported = true;
    F64 bandwidth = 0.0;
    U64 zeroReads = 0;
    U64 starts = 0;
    U64 stops = 0;
    bool streaming = false;
    U64 delivered = 0;
    std::vector<std::string> biasTee;
    std::size_t channels = 1;
    std::vector<std::string> antennaSets;
    std::string streamArguments;
    std::string streamFormat;
    bool timed = false;
    long long timeNs = 1000;
    long long gapNs = 0;
    U64 untimedReads = 0;
    bool abruptOnce = false;
    bool hardwareTime = false;
    long long hardwareLeadNs = 0;
    U64 overflows = 0;
    bool biasTeeRx = false;
    bool frozenTime = false;
    bool biasTeeRejected = false;
    bool singleChannelStreams = false;
    bool timedStartOnly = false;
    std::size_t streamChannels = 0;
    int startFlags = 0;
    F64 gainStep = 0.0;
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
char BiasTeeRxKey[] = "biastee_rx";
char AntennaA[] = "A";
char AntennaB[] = "B";
std::string ArgumentKey;
std::string ArgumentValue;

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
    api.getBandwidthRange = [](const Abi::Device*, int, std::size_t, std::size_t* count) {
        static Abi::Range range{0.2e6, 56.0e6, 0.0};
        *count = 1;
        return &range;
    };
    api.getSettingInfo = [](const Abi::Device*, std::size_t* count) {
        static Abi::ArgInfo setting{};
        setting.key = Fake().biasTeeRx ? BiasTeeRxKey : BiasTeeKey;
        *count = 1;
        return &setting;
    };
    api.writeSetting = [](Abi::Device*, const char* key, const char* value) {
        if (std::string(key) == (Fake().biasTeeRx ? "biastee_rx" : "biastee")) {
            if (Fake().biasTeeRejected) {
                return -1;
            }
            Fake().biasTee.emplace_back(value);
        }
        return 0;
    };
    api.kwargsFromString = [](const char* markup) {
        static char* keys[1];
        static char* values[1];
        const std::string text(markup ? markup : "");
        const auto split = text.find('=');
        if (split == std::string::npos) {
            return Abi::Kwargs{0, nullptr, nullptr};
        }
        ArgumentKey = text.substr(0, split);
        ArgumentValue = text.substr(split + 1);
        keys[0] = ArgumentKey.data();
        values[0] = ArgumentValue.data();
        return Abi::Kwargs{1, keys, values};
    };
    api.kwargsClear = [](Abi::Kwargs*) {};
    api.getNumChannels = [](const Abi::Device*, int) { return Fake().channels; };
    api.listAntennas = [](const Abi::Device*, int, std::size_t, std::size_t* count) -> char** {
        static char* names[] = {AntennaA, AntennaB};
        *count = Fake().channels > 1 ? 2 : 0;
        return *count > 0 ? names : nullptr;
    };
    api.getAntenna = [](const Abi::Device*, int, std::size_t) -> char* {
        return Fake().channels > 1 ? AntennaA : nullptr;
    };
    api.setAntenna = [](Abi::Device*, int, std::size_t channel, const char* name) {
        Fake().antennaSets.push_back(std::to_string(channel) + ":" + name);
        return 0;
    };
    api.getNativeStreamFormat = [](const Abi::Device*, int, std::size_t, double* fullScale)
        -> char* {
        static char native[] = "CS16";
        *fullScale = 2048.0;
        return native;
    };
    api.getStreamFormats = [](const Abi::Device*, int, std::size_t, std::size_t* count)
        -> char** {
        static char cf32[] = "CF32";
        static char cs16[] = "CS16";
        static char cs8[] = "CS8";
        static char* names[] = {cf32, cs16, cs8};
        *count = 3;
        return names;
    };
    api.hasGainMode = [](const Abi::Device*, int, std::size_t) { return true; };
    api.setGainMode = [](Abi::Device*, int, std::size_t, bool) { return 0; };
    api.getGainRange = [](const Abi::Device*, int, std::size_t) {
        return Abi::Range{0.0, 60.0, Fake().gainStep};
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
    api.setupStream = [](Abi::Device*, int, const char* format, const std::size_t*,
                         std::size_t count, const Abi::Kwargs* arguments) -> Abi::Stream* {
        if (Fake().singleChannelStreams && count > 1) {
            return nullptr;
        }
        Fake().streamFormat = format;
        Fake().streamChannels = count;
        Fake().streamArguments.clear();
        for (std::size_t i = 0; arguments && i < arguments->size; ++i) {
            Fake().streamArguments += std::string(arguments->keys[i]) + "=" + arguments->vals[i];
        }
        return reinterpret_cast<Abi::Stream*>(&Fake());
    };
    api.closeStream = [](Abi::Device*, Abi::Stream*) { return 0; };
    api.activateStream = [](Abi::Device*, Abi::Stream*, int flags, long long, std::size_t) {
        if (Fake().timedStartOnly && Fake().streamChannels > 1 && !(flags & Abi::HasTime)) {
            return -1;
        }
        Fake().startFlags = flags;
        ++Fake().starts;
        Fake().streaming = true;
        return 0;
    };
    api.deactivateStream = [](Abi::Device*, Abi::Stream*, int, long long) {
        ++Fake().stops;
        Fake().streaming = false;
        return 0;
    };
    api.readStream = [](Abi::Device*, Abi::Stream*, void* const*, std::size_t count, int* flags,
                        long long* timeNs, long) {
        auto& fake = Fake();
        *flags = 0;
        if (!fake.streaming) {
            return Abi::Timeout;
        }
        if (fake.overflows > 0) {
            --fake.overflows;
            return Abi::Overflow;
        }
        if (fake.timed) {
            fake.timeNs += fake.gapNs;
            fake.gapNs = 0;
            *timeNs = fake.frozenTime ? -250 : fake.timeNs;
            fake.timeNs += static_cast<long long>(count * 1.0e9 / fake.sampleRate);
            if (fake.untimedReads > 0) {
                --fake.untimedReads;
            } else {
                *flags |= Abi::HasTime;
            }
        }
        if (fake.abruptOnce) {
            fake.abruptOnce = false;
            *flags |= Abi::EndAbrupt;
        }
        if (fake.zeroReads > 0) {
            --fake.zeroReads;
            return 0;
        }
        fake.delivered += count;
        return static_cast<int>(count);
    };
    api.hasHardwareTime = [](const Abi::Device*, const char*) { return Fake().hardwareTime; };
    api.getHardwareTime = [](const Abi::Device*, const char*) {
        return Fake().timeNs + Fake().hardwareLeadNs;
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

TEST_CASE("SoapySDR sweep restarts the stream at each hop", "[devices][sdr][soapy]") {
    const FakeRuntime runtime;
    auto sweep = OpenSweep();

    CHECK(RunHops(*sweep, 3) == 3);
    CHECK(Fake().starts == 3);
    CHECK(Fake().stops == 2);
}

TEST_CASE("SoapySDR sweep drops samples received before each retune", "[devices][sdr][soapy]") {
    U64 plain = 0;
    {
        const FakeRuntime runtime({.timed = true});
        auto sweep = OpenSweep();
        CHECK(RunHops(*sweep, 2) == 2);
        plain = Fake().delivered;
    }

    const FakeRuntime runtime({.timed = true, .hardwareTime = true, .hardwareLeadNs = 99900});
    auto sweep = OpenSweep();
    CHECK(RunHops(*sweep, 2) == 2);
    CHECK(Fake().delivered == plain + 2 * 200);
}

TEST_CASE("SoapySDR sweep refills the hop after an overflow", "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.timed = true, .hardwareTime = true, .overflows = 1});
    auto sweep = OpenSweep();

    CHECK(RunHops(*sweep, 2) == 2);
    CHECK(sweep->statistics().failedHops == 0);
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
        CHECK(Fake().biasTee == std::vector<std::string>{"false", "true", "false"});
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

namespace {

std::unique_ptr<Radio::Stream> StartStream(const Radio::Stream::Settings& settings) {
    std::unique_ptr<Radio::Stream> stream;
    REQUIRE(Radio::Stream::Open("driver=soapy,serial=1234", stream) == Result::SUCCESS);
    REQUIRE(stream->start(settings) == Result::SUCCESS);
    return stream;
}

Radio::Device::Capture Read(Radio::Stream& stream) {
    std::vector<CF32> samples(256);
    return stream.read(std::as_writable_bytes(std::span(samples)), std::chrono::milliseconds(10));
}

}  // namespace

TEST_CASE("SoapySDR stream passes driver arguments and switches Bias-T off on close",
          "[devices][sdr][soapy]") {
    const FakeRuntime runtime;
    auto stream = StartStream({
        .frequency = 100.0e6,
        .sampleRate = 1.0e6,
        .biasTee = true,
        .driverArguments = "bufflen=8192",
    });

    CHECK(Fake().streamArguments == "bufflen=8192");
    REQUIRE_FALSE(Fake().biasTee.empty());
    CHECK(Fake().biasTee.back() == "true");

    stream.reset();
    CHECK(Fake().biasTee.back() == "false");
}

TEST_CASE("SoapySDR stream timestamps continue across reads and jump at a time gap",
          "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.timed = true});
    auto stream = StartStream({.frequency = 100.0e6, .sampleRate = 1.0e6, .settlingTime = 0.0});

    const auto first = Read(*stream);
    REQUIRE(first.status == Radio::Device::Status::Samples);
    REQUIRE(first.timestamp == 0);

    Fake().untimedReads = 1;
    const auto untimed = Read(*stream);
    REQUIRE(untimed.timestamp == 256);
    CHECK(untimed.epoch == first.epoch);

    Fake().gapNs = 5000;
    const auto gap = Read(*stream);
    CHECK(gap.timestamp == 517);
    CHECK(gap.epoch == first.epoch);

    REQUIRE(stream->update({.frequency = 200.0e6, .sampleRate = 1.0e6, .settlingTime = 0.0}) ==
            Result::SUCCESS);
    const auto retuned = Read(*stream);
    CHECK(retuned.timestamp == 0);
    CHECK(retuned.epoch > first.epoch);
    CHECK(Fake().starts == 2);
    CHECK(Fake().stops == 1);
}

TEST_CASE("SoapySDR stream sets a manual filter bandwidth or follows the sample rate",
          "[devices][sdr][soapy]") {
    const FakeRuntime runtime;
    auto stream = StartStream({.frequency = 100.0e6, .sampleRate = 1.0e6});
    CHECK(stream->capabilities().bandwidthRanges.size() == 1);
    CHECK(Fake().bandwidth == 1.0e6);

    REQUIRE(stream->update({.frequency = 100.0e6, .sampleRate = 1.0e6, .bandwidth = 0.6e6}) ==
            Result::SUCCESS);
    CHECK(Fake().bandwidth == 0.6e6);
}

TEST_CASE("SoapySDR stream offers the module's integer formats and opens the selected one",
          "[devices][sdr][soapy]") {
    const FakeRuntime runtime;
    auto stream = StartStream({.frequency = 100.0e6, .sampleRate = 1.0e6});
    CHECK(Fake().streamFormat == "CF32");
    const auto& caps = stream->capabilities();
    REQUIRE(caps.formats.size() == 3);
    CHECK(caps.format(DataType::CI16)->fullScale == 2048.0);
    CHECK(caps.format(DataType::CI8)->fullScale == 128.0);
    CHECK_FALSE(caps.format(DataType::CU8));

    REQUIRE(stream->update({.frequency = 100.0e6, .sampleRate = 1.0e6, .settlingTime = 0.0,
                            .format = DataType::CI16}) == Result::SUCCESS);
    CHECK(Fake().streamFormat == "CS16");
    std::vector<CI16> samples(256);
    const auto capture = stream->read(std::as_writable_bytes(std::span(samples)),
                                      std::chrono::milliseconds(10));
    CHECK(capture.status == Radio::Device::Status::Samples);
    CHECK(capture.count == samples.size());

    CHECK(stream->update({.frequency = 100.0e6, .sampleRate = 1.0e6,
                          .format = DataType::CU8}) != Result::SUCCESS);
}

TEST_CASE("SoapySDR stream drives Bias-T through the RX-only setting", "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.biasTeeRx = true});
    auto stream = StartStream({.frequency = 100.0e6, .sampleRate = 1.0e6, .biasTee = true});

    CHECK(stream->capabilities().biasTee);
    CHECK(stream->applied().biasTee);
    REQUIRE_FALSE(Fake().biasTee.empty());
    CHECK(Fake().biasTee.back() == "true");
}

TEST_CASE("SoapySDR stream hides a Bias-T setting the driver rejects", "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.biasTeeRx = true, .biasTeeRejected = true});
    auto stream = StartStream({.frequency = 100.0e6, .sampleRate = 1.0e6, .biasTee = true});

    CHECK_FALSE(stream->capabilities().biasTee);
    CHECK_FALSE(stream->applied().biasTee);
}

TEST_CASE("SoapySDR stream drops samples received before a retune", "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.timed = true, .hardwareTime = true, .hardwareLeadNs = 100000});
    auto stream = StartStream({.frequency = 100.0e6, .sampleRate = 1.0e6, .settlingTime = 0.0});
    REQUIRE(Read(*stream).count == 256);

    REQUIRE(stream->update({.frequency = 200.0e6, .sampleRate = 1.0e6, .settlingTime = 0.0}) ==
            Result::SUCCESS);
    const auto retuned = Read(*stream);
    REQUIRE(retuned.status == Radio::Device::Status::Samples);
    CHECK(retuned.count == 156);
    CHECK(retuned.timestamp == 100);
    CHECK(Read(*stream).count == 256);
    CHECK(Fake().starts == 2);
}

TEST_CASE("SoapySDR stream settles by sample count when reads never carry timestamps",
          "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.hardwareTime = true, .hardwareLeadNs = 599500});
    auto stream = StartStream({.frequency = 100.0e6, .sampleRate = 1.0e6, .settlingTime = 0.0});
    REQUIRE(Read(*stream).count == 256);

    REQUIRE(stream->update({.frequency = 200.0e6, .sampleRate = 1.0e6, .settlingTime = 0.0}) ==
            Result::SUCCESS);
    CHECK(Read(*stream).count == 256);
}

TEST_CASE("SoapySDR stream keeps the retune cutoff across untimed reads", "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.timed = true, .hardwareTime = true, .hardwareLeadNs = 599500});
    auto stream = StartStream({.frequency = 100.0e6, .sampleRate = 1.0e6, .settlingTime = 0.0});
    REQUIRE(Read(*stream).count == 256);

    REQUIRE(stream->update({.frequency = 200.0e6, .sampleRate = 1.0e6, .settlingTime = 0.0}) ==
            Result::SUCCESS);
    CHECK(Read(*stream).status == Radio::Device::Status::Timeout);
    Fake().untimedReads = 1;
    CHECK(Read(*stream).status == Radio::Device::Status::Timeout);
    CHECK(Read(*stream).count == 168);
}

TEST_CASE("SoapySDR stream settles by sample count when timestamps do not advance",
          "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.timed = true, .hardwareTime = true, .frozenTime = true});
    auto stream = StartStream({.frequency = 100.0e6, .sampleRate = 1.0e6, .settlingTime = 0.0});
    REQUIRE(Read(*stream).count == 256);

    REQUIRE(stream->update({.frequency = 200.0e6, .sampleRate = 1.0e6, .settlingTime = 0.0}) ==
            Result::SUCCESS);
    CHECK(Read(*stream).status == Radio::Device::Status::Timeout);
    CHECK(Read(*stream).count == 256);
}

TEST_CASE("SoapySDR stream reports a gap after an abrupt end", "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.abruptOnce = true});
    auto stream = StartStream({.frequency = 100.0e6, .sampleRate = 1.0e6, .settlingTime = 0.0});

    CHECK(Read(*stream).status == Radio::Device::Status::Samples);
    CHECK(Read(*stream).status == Radio::Device::Status::Discontinuity);
    CHECK(Read(*stream).status == Radio::Device::Status::Samples);
}

TEST_CASE("SoapySDR stream sets the antenna of each receiver", "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.channels = 2});
    auto stream = StartStream({
        .frequency = 100.0e6,
        .sampleRate = 1.0e6,
        .automaticGain = false,
        .receiveChannels = 2,
        .channelGains = {30.0, 30.0},
        .channelAntennas = {"B", "A"},
    });
    CHECK(Fake().antennaSets == std::vector<std::string>{"0:B", "1:A"});

    Fake().antennaSets.clear();
    REQUIRE(stream->update({
        .frequency = 100.0e6,
        .sampleRate = 1.0e6,
        .automaticGain = false,
        .receiveChannels = 2,
        .channelGains = {30.0, 30.0},
        .channelAntennas = {"", "A"},
    }) == Result::SUCCESS);
    CHECK(Fake().antennaSets == std::vector<std::string>{"0:A", "1:A"});
}

TEST_CASE("SoapySDR stream offers one receiver when the driver streams one at a time",
          "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.channels = 2, .singleChannelStreams = true});
    std::unique_ptr<Radio::Stream> stream;
    REQUIRE(Radio::Stream::Open("driver=soapy,serial=1234", stream) == Result::SUCCESS);
    CHECK(stream->capabilities().receiveChannels == 1);
}

TEST_CASE("SoapySDR stream starts several receivers at a hardware time when required",
          "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.channels = 2, .hardwareTime = true, .timedStartOnly = true});
    auto stream = StartStream({
        .frequency = 100.0e6,
        .sampleRate = 1.0e6,
        .automaticGain = false,
        .receiveChannels = 2,
        .channelGains = {30.0, 30.0},
    });
    CHECK((Fake().startFlags & Abi::HasTime) != 0);
}

TEST_CASE("SoapySDR stream applies manual gain on the driver gain step", "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.gainStep = 1.0});
    auto stream = StartStream({
        .frequency = 100.0e6,
        .sampleRate = 1.0e6,
        .automaticGain = false,
        .gain = 30.4,
    });
    CHECK(stream->applied().gain == 30.0);
}

TEST_CASE("SoapySDR stream leaves the driver antenna alone on Default", "[devices][sdr][soapy]") {
    const FakeRuntime runtime({.channels = 2});
    auto stream = StartStream({
        .frequency = 100.0e6,
        .sampleRate = 1.0e6,
        .automaticGain = false,
        .receiveChannels = 2,
        .channelGains = {30.0, 30.0},
        .channelAntennas = {"", ""},
    });
    CHECK(Fake().antennaSets.empty());
}

#endif
