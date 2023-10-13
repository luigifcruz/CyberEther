#include "jetstream/logger.hh"

std::string& JST_LOG_LAST_WARNING() {
    static std::string _JST_LOG_LAST_WARNING;
    return _JST_LOG_LAST_WARNING;
}

std::string& JST_LOG_LAST_ERROR() {
    static std::string _JST_LOG_LAST_ERROR;
    return _JST_LOG_LAST_ERROR;
}

std::string& JST_LOG_LAST_FATAL() {
    static std::string _JST_LOG_LAST_FATAL;
    return _JST_LOG_LAST_FATAL;
}