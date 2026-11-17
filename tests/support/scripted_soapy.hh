#ifndef JETSTREAM_TESTS_SUPPORT_SCRIPTED_SOAPY_HH
#define JETSTREAM_TESTS_SUPPORT_SCRIPTED_SOAPY_HH

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
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
    F64 sampleRateOffset = 0.0;
    std::optional<F64> nextSampleRateReadback;
    std::atomic<int> status{0};
    std::atomic<U64> sampleRateWrites{0};
    std::atomic<U64> gainModeWrites{0};
    std::atomic<bool> emitSamples{false};
    std::atomic<unsigned> activeReads{0};
    std::atomic<U64> concurrentChanges{0};

    void reset() {
        std::lock_guard lock(mutex);
        applied = {};
        failure.clear();
        automaticGainSupported = true;
        sampleRateOffset = 0.0;
        nextSampleRateReadback.reset();
        status = 0;
        sampleRateWrites = 0;
        gainModeWrites = 0;
        emitSamples = false;
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

    F64 readSampleRate() {
        std::lock_guard lock(mutex);
        if (activeReads > 0) {
            ++concurrentChanges;
        }
        status = 0;
        if (failure == "sampleRateReadback") {
            failure.clear();
            status = -1;
        }
        const F64 value = nextSampleRateReadback.value_or(applied.sampleRate);
        nextSampleRateReadback.reset();
        return value;
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
    api.lastStatus = [] { return State().status.load(); };
    api.free = [](void*) {};
    api.stringsClear = [](char***, std::size_t) {};
    api.kwargsFromString = [](const char*) { return SoapyAbi::Kwargs{}; };
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
        ++State().sampleRateWrites;
        return State().set("sampleRate", &Settings::sampleRate, value + State().sampleRateOffset);
    };
    api.getSampleRate = [](const SoapyAbi::Device*, int, std::size_t) {
        return State().readSampleRate();
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
    api.setupStream = [](SoapyAbi::Device*, int, const char*, const std::size_t*,
                          std::size_t, const SoapyAbi::Kwargs*) {
        return reinterpret_cast<SoapyAbi::Stream*>(&State());
    };
    api.activateStream = [](SoapyAbi::Device*, SoapyAbi::Stream*, int, long long, std::size_t) {
        return 0;
    };
    api.deactivateStream = [](SoapyAbi::Device*, SoapyAbi::Stream*, int, long long) { return 0; };
    api.closeStream = [](SoapyAbi::Device*, SoapyAbi::Stream*) { return 0; };
    api.readStream = [](SoapyAbi::Device*, SoapyAbi::Stream*, void* const* buffers,
                         std::size_t count, int*, long long*, long) {
        ++State().activeReads;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        int result = SoapyAbi::Timeout;
        if (State().emitSamples) {
            std::fill_n(static_cast<CF32*>(buffers[0]), count, CF32{1.0f, 0.0f});
            result = static_cast<int>(count);
        }
        --State().activeReads;
        return result;
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
