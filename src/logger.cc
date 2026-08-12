// jetstream/logger.cc
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <ostream>
#include <string>

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

namespace {

int InitialDebugLevel() {
    std::string value;
    if (Jetstream::Platform::EnvironmentVariable("JST_DEBUG", value) != Jetstream::Result::SUCCESS) {
        return JST_LOG_DEBUG_DEFAULT_LEVEL;
    }

    return std::atoi(value.c_str());
}

bool InitialColorEnabled() {
    std::string value;
    if (Jetstream::Platform::EnvironmentVariable("NO_COLOR", value) ==
            Jetstream::Result::SUCCESS &&
        !value.empty()) {
        return false;
    }
    if (Jetstream::Platform::EnvironmentVariable("TERM", value) ==
            Jetstream::Result::SUCCESS &&
        value == "dumb") {
        return false;
    }

    return Jetstream::Platform::PrepareStandardOutputForAnsi();
}

}  // namespace

std::string& JST_LOG_LAST_WARNING() {
    static std::string __JST_LOG_LAST_WARNING;
    return __JST_LOG_LAST_WARNING;
}

std::string& JST_LOG_LAST_ERROR() {
    static std::string __JST_LOG_LAST_ERROR;
    return __JST_LOG_LAST_ERROR;
}

std::string& JST_LOG_LAST_FATAL() {
    static std::string __JST_LOG_LAST_FATAL;
    return __JST_LOG_LAST_FATAL;
}

std::mutex& _JST_LOG_MUTEX() {
    static std::mutex __JST_LOG_MUTEX;
    return __JST_LOG_MUTEX;
}

int& _JST_LOG_DEBUG_LEVEL() {
    static int __JST_LOG_DEBUG_LEVEL = InitialDebugLevel();
    return __JST_LOG_DEBUG_LEVEL;
}

void JST_LOG_SET_DEBUG_LEVEL(int level) {
    _JST_LOG_DEBUG_LEVEL() = level;
}

namespace {
    std::atomic<std::ostream*> g_sink{&std::cout};
    std::atomic<int> g_color_override{-1};

    bool ColorEnabled() {
        const int colorOverride = g_color_override.load(std::memory_order_acquire);
        if (colorOverride >= 0) {
            return colorOverride != 0;
        }

        static const bool enabled = InitialColorEnabled();
        return enabled;
    }
}

std::ostream& JST_LOG_SINK() {
    return *g_sink.load(std::memory_order_acquire);
}

void JST_LOG_SET_SINK(std::ostream* s) {
    g_sink.store(s ? s : &std::cout, std::memory_order_release);
}

void JST_LOG_RESTORE_STDOUT() {
    g_sink.store(&std::cout, std::memory_order_release);
}

void JST_LOG_COLOR(bool enable) {
    if (enable) {
        (void)Jetstream::Platform::PrepareStandardOutputForAnsi();
    }
    g_color_override.store(enable ? 1 : 0, std::memory_order_release);
}

jst::fmt::text_style JST_LOG_STYLE(const jst::fmt::text_style& ts) {
#if defined(JST_OS_IOS) || defined(JST_OS_BROWSER)
    return jst::fmt::text_style();
#endif
    if (ColorEnabled()) {
        return ts;
    }
    return jst::fmt::text_style();
}
