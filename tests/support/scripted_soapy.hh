#ifndef JETSTREAM_TESTS_SUPPORT_SCRIPTED_SOAPY_HH
#define JETSTREAM_TESTS_SUPPORT_SCRIPTED_SOAPY_HH

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <type_traits>

#include "domains/io/soapy/library/soapysdr.hh"

namespace TestSoapy {

using namespace Jetstream;
using namespace Jetstream::Modules;

struct Settings {
    F64 frequency = 100.0e6;
    F64 sampleRate = 2.0e6;
    bool automaticGain = false;
    F64 gain = 30.0;
    F64 bandwidth = 8.0e6;
};

struct Script {
    std::mutex mutex;
    Settings applied;
    std::string failure;
    std::atomic<int> queryStatus{0};
    std::atomic<U64> queries{0};
    F64 sampleRateStep = 0.0;
    F64 frequencyStep = 0.0;
    std::vector<SoapyAbi::Range> sampleRateRanges{{1.0e6, 10.0e6, 0.0}};
    std::vector<SoapyAbi::Range> frequencyRanges{{10.0e6, 2000.0e6, 0.0}};
    SoapyAbi::Range gainRange{0.0, 60.0, 0.0};
    bool controlsSupported = false;
    std::map<std::string, F64> gainStages{{"LNA", 10.0}, {"VGA", 20.0}};
    std::map<std::string, std::string> driverSettings{
        {"enabled", "false"}, {"decimation", "2"}, {"correction", "0.5"},
        {"mode", "fast"}, {"label", "original"}};
    SoapyKwargs streamArgs;
    U64 totalGainWrites = 0;
    U64 bandwidthWrites = 0;
    bool streamStarted = false;
    bool automaticGainSupported = true;
    bool biasTeeSupported = false;
    bool biasTeeEnabled = false;
    std::vector<std::string> streamFormats{"CF32", "CF64", "CS16", "CS8", "CU8"};
    std::string nativeFormat = "CS16";
    F64 fullScale = 32768.0;
    std::string streamFormat;
    std::vector<char*> formatPointers;
    std::vector<I8> samples;
    std::deque<int> reads;
    std::size_t sampleBytes = sizeof(CF32);
    std::size_t position = 0;
    std::atomic<U64> gainModeWrites{0};
    std::atomic<unsigned> activeReads{0};
    std::atomic<U64> concurrentChanges{0};

    void reset() {
        std::lock_guard lock(mutex);
        applied = {};
        failure.clear();
        queryStatus = 0;
        queries = 0;
        sampleRateStep = 0.0;
        frequencyStep = 0.0;
        sampleRateRanges = {{1.0e6, 10.0e6, 0.0}};
        frequencyRanges = {{10.0e6, 2000.0e6, 0.0}};
        gainRange = {0.0, 60.0, 0.0};
        controlsSupported = false;
        gainStages = {{"LNA", 10.0}, {"VGA", 20.0}};
        driverSettings = {{"enabled", "false"}, {"decimation", "2"},
                          {"correction", "0.5"}, {"mode", "fast"}, {"label", "original"}};
        streamArgs.clear();
        totalGainWrites = 0;
        bandwidthWrites = 0;
        streamStarted = false;
        automaticGainSupported = true;
        biasTeeSupported = false;
        biasTeeEnabled = false;
        streamFormats = {"CF32", "CF64", "CS16", "CS8", "CU8"};
        nativeFormat = "CS16";
        fullScale = 32768.0;
        streamFormat.clear();
        formatPointers.clear();
        samples.clear();
        reads.clear();
        sampleBytes = sizeof(CF32);
        position = 0;
        gainModeWrites = 0;
        activeReads = 0;
        concurrentChanges = 0;
    }

    Settings settings() {
        std::lock_guard lock(mutex);
        return applied;
    }

    void fail(const std::string& call) {
        std::lock_guard lock(mutex);
        failure = call;
    }

    void query(const char* name) {
        std::lock_guard lock(mutex);
        ++queries;
        if (activeReads > 0) {
            ++concurrentChanges;
        }
        queryStatus = failure == name ? -1 : 0;
        if (queryStatus != 0) {
            failure.clear();
        }
    }

    template<typename T>
    T get(const char* name, T Settings::* field) {
        query(name);
        return settings().*field;
    }

    template<typename T>
    int set(const char* name, T Settings::* field, T value) {
        std::lock_guard lock(mutex);
        if (activeReads > 0) {
            ++concurrentChanges;
        }
        if (failure == name) {
            failure.clear();
            return -1;
        }
        if constexpr (std::is_same_v<T, F64>) {
            const F64 step = field == &Settings::sampleRate ? sampleRateStep
                           : field == &Settings::frequency ? frequencyStep : 0.0;
            if (step > 0.0) {
                value = std::round(value / step) * step;
            }
        }
        applied.*field = value;
        return 0;
    }
};

inline Script& State() {
    static Script script;
    return script;
}

inline SoapyApi Api() {
    SoapyApi api;
    api.lastError = [] { return "injected failure"; };
    api.lastStatus = [] { return State().queryStatus.load(); };
    api.errToStr = [](int) { return "injected error"; };
    api.free = [](void*) {};
    api.stringsClear = [](char***, std::size_t) {};
    api.kwargsFromString = [](const char*) { return SoapyAbi::Kwargs{}; };
    api.kwargsToString = [](const SoapyAbi::Kwargs*) {
        static char args[] = "driver=scripted";
        return args;
    };
    api.kwargsClear = [](SoapyAbi::Kwargs*) {};
    api.kwargsListClear = [](SoapyAbi::Kwargs*, std::size_t) {};
    api.argInfoListClear = [](SoapyAbi::ArgInfo*, std::size_t) {};
    api.enumerate = [](const SoapyAbi::Kwargs*, std::size_t* count) {
        static SoapyAbi::Kwargs entry{};
        *count = 1;
        return &entry;
    };
    api.make = [](const SoapyAbi::Kwargs*) {
        return reinterpret_cast<SoapyAbi::Device*>(&State());
    };
    api.unmake = [](SoapyAbi::Device*) { return 0; };
    api.getSampleRateRange = [](const SoapyAbi::Device*, int, std::size_t,
                                 std::size_t* count) {
        State().query("sampleRateRange");
        *count = State().sampleRateRanges.size();
        return State().sampleRateRanges.data();
    };
    api.getFrequencyRange = [](const SoapyAbi::Device*, int, std::size_t,
                                std::size_t* count) {
        State().query("frequencyRange");
        *count = State().frequencyRanges.size();
        return State().frequencyRanges.data();
    };
    api.getSettingInfo = [](const SoapyAbi::Device*, std::size_t* count)
        -> SoapyAbi::ArgInfo* {
        State().query("settings");
        static std::vector<SoapyAbi::ArgInfo> args;
        args.clear();
        const auto argument = [](const char* key, int type, const char* value,
                                  SoapyAbi::Range range = {}) {
            SoapyAbi::ArgInfo info{};
            info.key = const_cast<char*>(key);
            info.name = const_cast<char*>(key);
            info.value = const_cast<char*>(value);
            info.type = type;
            info.range = range;
            return info;
        };
        if (State().controlsSupported) {
            args = {argument("enabled", 0, "false"),
                    argument("decimation", 1, "2", {1.0, 8.0, 1.0}),
                    argument("correction", 2, "0.5", {-1.0, 1.0, 0.25}),
                    argument("mode", 3, "fast"), argument("label", 3, "original")};
            static char* options[]{const_cast<char*>("fast"), const_cast<char*>("quiet")};
            static char* names[]{const_cast<char*>("Fast"), const_cast<char*>("Quiet")};
            args[3].numOptions = 2;
            args[3].options = options;
            args[3].optionNames = names;
        }
        if (State().biasTeeSupported) {
            args.push_back(argument("biastee", 0, "false"));
        }
        *count = args.size();
        return args.data();
    };
    api.writeSetting = [](SoapyAbi::Device*, const char* key, const char* value) {
        auto& state = State();
        std::lock_guard lock(state.mutex);
        if (std::string(key) == "biastee") {
            state.biasTeeEnabled = std::string(value) == "true";
            return 0;
        }
        if (state.streamStarted) {
            ++state.concurrentChanges;
        }
        if (state.failure == std::string("setting:") + key) {
            state.failure.clear();
            return -1;
        }
        state.driverSettings[key] = value;
        return 0;
    };
    api.readSetting = [](const SoapyAbi::Device*, const char* key) {
        State().query("readSetting");
        return State().driverSettings[key].data();
    };
    api.getStreamArgsInfo = [](const SoapyAbi::Device*, int, std::size_t,
                               std::size_t* count) -> SoapyAbi::ArgInfo* {
        State().query("streamArgs");
        static char* options[]{const_cast<char*>("sc16"), const_cast<char*>("sc8")};
        static SoapyAbi::ArgInfo wire{};
        wire.key = const_cast<char*>("wire");
        wire.value = const_cast<char*>("sc16");
        wire.type = 3;
        wire.numOptions = 2;
        wire.options = options;
        *count = State().controlsSupported ? 1 : 0;
        return &wire;
    };
    api.listGains = [](const SoapyAbi::Device*, int, std::size_t, std::size_t* count) {
        State().query("gains");
        static char* names[]{const_cast<char*>("LNA"), const_cast<char*>("VGA")};
        *count = State().controlsSupported ? 2 : 0;
        return names;
    };
    api.getGainElementRange = [](const SoapyAbi::Device*, int, std::size_t,
                                 const char* name) {
        State().query("gainRange");
        return SoapyAbi::Range{0.0, 30.0, std::string(name) == "LNA" ? 1.0 : 0.5};
    };
    api.getGainElement = [](const SoapyAbi::Device*, int, std::size_t, const char* name) {
        State().query("gainElement");
        std::lock_guard lock(State().mutex);
        return State().gainStages.at(name);
    };
    api.setGainElement = [](SoapyAbi::Device*, int, std::size_t, const char* name,
                            double value) {
        auto& state = State();
        std::lock_guard lock(state.mutex);
        if (state.activeReads > 0) {
            ++state.concurrentChanges;
        }
        if (state.failure == std::string("gain:") + name) {
            state.failure.clear();
            return -1;
        }
        state.gainStages.at(name) = value;
        state.applied.gain = state.gainStages.at("LNA") + state.gainStages.at("VGA");
        return 0;
    };
    api.getBandwidthRange = [](const SoapyAbi::Device*, int, std::size_t,
                                std::size_t* count) {
        State().query("bandwidthRange");
        static SoapyAbi::Range range{1.0e6, 20.0e6, 1.0e6};
        *count = State().controlsSupported ? 1 : 0;
        return &range;
    };
    api.getBandwidth = [](const SoapyAbi::Device*, int, std::size_t) {
        return State().get("getBandwidth", &Settings::bandwidth);
    };
    api.setBandwidth = [](SoapyAbi::Device*, int, std::size_t, double value) {
        ++State().bandwidthWrites;
        return State().set("bandwidth", &Settings::bandwidth,
                            std::round(value / 1.0e6) * 1.0e6);
    };
    api.listAntennas = [](const SoapyAbi::Device*, int, std::size_t, std::size_t* count)
        -> char** {
        State().query("antennas");
        *count = 0;
        return nullptr;
    };
    api.getAntenna = [](const SoapyAbi::Device*, int, std::size_t) {
        State().query("getAntenna");
        static char antenna[] = "RX";
        return antenna;
    };
    api.getGainRange = [](const SoapyAbi::Device*, int, std::size_t) {
        State().query("gainRange");
        return State().gainRange;
    };
    api.getFrequency = [](const SoapyAbi::Device*, int, std::size_t) {
        return State().get("getFrequency", &Settings::frequency);
    };
    api.getSampleRate = [](const SoapyAbi::Device*, int, std::size_t) {
        return State().get("getSampleRate", &Settings::sampleRate);
    };
    api.getGain = [](const SoapyAbi::Device*, int, std::size_t) {
        return State().get("getGain", &Settings::gain);
    };
    api.getGainMode = [](const SoapyAbi::Device*, int, std::size_t) {
        return State().get("getGainMode", &Settings::automaticGain);
    };
    api.setFrequency = [](SoapyAbi::Device*, int, std::size_t, double value, const SoapyAbi::Kwargs*) {
        return State().set("frequency", &Settings::frequency, value);
    };
    api.setSampleRate = [](SoapyAbi::Device*, int, std::size_t, double value) {
        return State().set("sampleRate", &Settings::sampleRate, value);
    };
    api.hasGainMode = [](const SoapyAbi::Device*, int, std::size_t) {
        State().query("gainModeSupport");
        return State().automaticGainSupported;
    };
    api.setGainMode = [](SoapyAbi::Device*, int, std::size_t, bool value) {
        ++State().gainModeWrites;
        if (!State().automaticGainSupported) {
            return -1;
        }
        return State().set("automaticGain", &Settings::automaticGain, value);
    };
    api.setGain = [](SoapyAbi::Device*, int, std::size_t, double value) {
        ++State().totalGainWrites;
        return State().set("manualGain", &Settings::gain, value);
    };
    api.getStreamFormats = [](const SoapyAbi::Device*, int, std::size_t,
                              std::size_t* count) {
        auto& state = State();
        state.query("formats");
        state.formatPointers.clear();
        for (auto& format : state.streamFormats) {
            state.formatPointers.push_back(format.data());
        }
        *count = state.formatPointers.size();
        return state.formatPointers.data();
    };
    api.getNativeStreamFormat = [](const SoapyAbi::Device*, int, std::size_t,
                                    double* fullScale) {
        State().query("nativeFormat");
        *fullScale = State().fullScale;
        return State().nativeFormat.data();
    };
    api.setupStream = [](SoapyAbi::Device*, int, const char* format, const std::size_t*,
                          std::size_t, const SoapyAbi::Kwargs* args) {
        State().streamArgs.clear();
        for (std::size_t i = 0; i < args->size; ++i) {
            State().streamArgs[args->keys[i]] = args->vals[i];
        }
        State().streamFormat = format;
        State().position = 0;
        for (const auto& supported : SoapySampleFormats) {
            if (State().streamFormat == supported.name) {
                State().sampleBytes = DataTypeSize(supported.dtype);
            }
        }
        return reinterpret_cast<SoapyAbi::Stream*>(&State());
    };
    api.activateStream = [](SoapyAbi::Device*, SoapyAbi::Stream*, int, long long, std::size_t) {
        State().streamStarted = true;
        return 0;
    };
    api.deactivateStream = [](SoapyAbi::Device*, SoapyAbi::Stream*, int, long long) { return 0; };
    api.closeStream = [](SoapyAbi::Device*, SoapyAbi::Stream*) {
        State().streamStarted = false;
        return 0;
    };
    api.readStream = [](SoapyAbi::Device*, SoapyAbi::Stream*, void* const* buffers,
                         std::size_t count, int*,
                         long long*, long) {
        auto& state = State();
        ++state.activeReads;
        struct ReadFinished {
            ~ReadFinished() { --State().activeReads; }
        } finished;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::lock_guard lock(state.mutex);
        if (!state.reads.empty()) {
            const int next = state.reads.front();
            state.reads.pop_front();
            if (next <= 0) {
                return next;
            }
            count = std::min(count, static_cast<std::size_t>(next));
        }
        const auto remaining =
            state.samples.size() / state.sampleBytes - state.position;
        count = std::min(count, remaining);
        if (count == 0) {
            return SoapyAbi::Timeout;
        }
        const auto* source = state.samples.data() + state.position * state.sampleBytes;
        std::memcpy(buffers[0], source,
                    count * state.sampleBytes);
        state.position += count;
        return static_cast<int>(count);
    };
    return api;
}

struct Runtime {
    Runtime() {
        State().reset();
        Modules::SoapyRuntime::Override(Api());
        Modules::SoapyDiscovery::ClearCache();
    }

    ~Runtime() {
        Modules::SoapyRuntime::Reset();
        Modules::SoapyDiscovery::ClearCache();
    }
};

}

#endif
