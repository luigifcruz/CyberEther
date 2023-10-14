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