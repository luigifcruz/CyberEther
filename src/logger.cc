// jetstream/logger.cc
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <ostream>
#include <string>

#include "jetstream/logger.hh"
#include "fmt/color.h"

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
    static int __JST_LOG_DEBUG_LEVEL =
        (std::getenv("JST_DEBUG") ? std::atoi(std::getenv("JST_DEBUG"))
                                  : JST_LOG_DEBUG_DEFAULT_LEVEL);
    return __JST_LOG_DEBUG_LEVEL;
}

void JST_LOG_SET_DEBUG_LEVEL(int level) {
    _JST_LOG_DEBUG_LEVEL() = level;
}

namespace {
    std::atomic<bool> g_color_enabled{true};
    std::atomic<std::ostream*> g_sink{&std::cout};
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
    g_color_enabled.store(enable, std::memory_order_release);
}

jst::fmt::text_style JST_LOG_STYLE(const jst::fmt::text_style& ts) {
#if defined(JST_OS_IOS) or defined(JST_OS_BROWSER)
    return jst::fmt::text_style();
#endif
    if (g_color_enabled.load(std::memory_order_acquire)) {
        return ts;
    }
    return jst::fmt::text_style();
}
