#ifndef JETSTREAM_TESTS_SUPPORT_SCRIPTED_SOAPY_HH
#define JETSTREAM_TESTS_SUPPORT_SCRIPTED_SOAPY_HH

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

#include "domains/io/soapy/library/soapysdr.hh"

namespace TestSoapy {

using namespace Jetstream;
using namespace Jetstream::Modules;

struct Settings {
    F64 frequency = 100.0e6;
    F64 sampleRate = 2.0e6;
    bool automaticGain = false;
    F64 gain = 30.0;
};

struct Script {
    std::mutex mutex;
    Settings applied;
    std::string failure;
    bool automaticGainSupported = true;
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
        automaticGainSupported = true;
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

    template<typename T>
    int set(const char* name, T Settings::* field, const T value) {
        std::lock_guard lock(mutex);
        if (activeReads > 0) {
            ++concurrentChanges;
        }
        if (failure == name) {
            failure.clear();
            return -1;
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
    api.lastStatus = [] { return 0; };
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
    api.getSampleRateRange = [](const SoapyAbi::Device*, int, std::size_t, std::size_t* count) {
        static SoapyAbi::Range range{1.0e6, 10.0e6, 0.0};
        *count = 1;
        return &range;
    };
    api.getFrequencyRange = [](const SoapyAbi::Device*, int, std::size_t, std::size_t* count) {
        static SoapyAbi::Range range{10.0e6, 2000.0e6, 0.0};
        *count = 1;
        return &range;
    };
    api.getSettingInfo = [](const SoapyAbi::Device*, std::size_t* count) -> SoapyAbi::ArgInfo* {
        *count = 0;
        return nullptr;
    };
    api.listAntennas = [](const SoapyAbi::Device*, int, std::size_t, std::size_t* count) -> char** {
        *count = 0;
        return nullptr;
    };
    api.getGainRange = [](const SoapyAbi::Device*, int, std::size_t) {
        return SoapyAbi::Range{0.0, 60.0, 0.0};
    };
    api.setFrequency = [](SoapyAbi::Device*, int, std::size_t, double value, const SoapyAbi::Kwargs*) {
        return State().set("frequency", &Settings::frequency, value);
    };
    api.setSampleRate = [](SoapyAbi::Device*, int, std::size_t, double value) {
        return State().set("sampleRate", &Settings::sampleRate, value);
    };
    api.hasGainMode = [](const SoapyAbi::Device*, int, std::size_t) {
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
        return State().set("manualGain", &Settings::gain, value);
    };
    api.getStreamFormats = [](const SoapyAbi::Device*, int, std::size_t,
                              std::size_t* count) {
        auto& state = State();
        state.formatPointers.clear();
        for (auto& format : state.streamFormats) {
            state.formatPointers.push_back(format.data());
        }
        *count = state.formatPointers.size();
        return state.formatPointers.data();
    };
    api.getNativeStreamFormat = [](const SoapyAbi::Device*, int, std::size_t,
                                    double* fullScale) {
        *fullScale = State().fullScale;
        return State().nativeFormat.data();
    };
    api.setupStream = [](SoapyAbi::Device*, int, const char* format, const std::size_t*,
                          std::size_t, const SoapyAbi::Kwargs*) {
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
        return 0;
    };
    api.deactivateStream = [](SoapyAbi::Device*, SoapyAbi::Stream*, int, long long) { return 0; };
    api.closeStream = [](SoapyAbi::Device*, SoapyAbi::Stream*) { return 0; };
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
