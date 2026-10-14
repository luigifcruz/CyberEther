#ifndef JETSTREAM_TESTS_SUPPORT_SCRIPTED_SOAPY_HH
#define JETSTREAM_TESTS_SUPPORT_SCRIPTED_SOAPY_HH

#include <atomic>
#include <chrono>
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
    std::atomic<unsigned> activeReads{0};
    std::atomic<U64> concurrentChanges{0};

    void reset() {
        std::lock_guard lock(mutex);
        applied = {};
        failure.clear();
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
    api.setGainMode = [](SoapyAbi::Device*, int, std::size_t, bool value) {
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
    api.readStream = [](SoapyAbi::Device*, SoapyAbi::Stream*, void* const*, std::size_t, int*,
                         long long*, long) {
        ++State().activeReads;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        --State().activeReads;
        return SoapyAbi::Timeout;
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
