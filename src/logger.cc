#include <mutex>
#include <cstdlib>
#include <string>

#include "jetstream/logger.hh"

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

void JST_LOG_SET_DEBUG_LEVEL(int level) {
    _JST_LOG_DEBUG_LEVEL() = level;
}

int& _JST_LOG_DEBUG_LEVEL() {
    static int __JST_LOG_DEBUG_LEVEL = (getenv("JST_DEBUG") ? \
                                            std::atoi(getenv("JST_DEBUG")) : \
                                            JST_LOG_DEBUG_DEFAULT_LEVEL);
    return __JST_LOG_DEBUG_LEVEL;
}