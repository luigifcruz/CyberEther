#ifndef JETSTREAM_LOGGER_HH
#define JETSTREAM_LOGGER_HH

#include <iostream>

#include <fmt/format.h>
#include <fmt/color.h>

#define _JST_LOG_SINK          std::cout
#define _JST_LOG_ENDL          std::endl;

#define _JST_LOG_BOLD(...)     fmt::format(fmt::emphasis::bold, __VA_ARGS__)
#define _JST_LOG_DEFAULT(...)  fmt::format(__VA_ARGS__)
#define _JST_LOG_ORANGE(...)   fmt::format(fmt::fg(fmt::color::orange), __VA_ARGS__)
#define _JST_LOG_YELLOW(...)   fmt::format(fmt::fg(fmt::color::yellow), __VA_ARGS__)
#define _JST_LOG_CYAN(...)     fmt::format(fmt::fg(fmt::color::aqua), __VA_ARGS__)
#define _JST_LOG_RED(...)      fmt::format(fmt::fg(fmt::color::red), __VA_ARGS__)
#define _JST_LOG_MAGENTA(...)  fmt::format(fmt::fg(fmt::color::fuchsia), __VA_ARGS__)

#define _JST_LOG_NAME          _JST_LOG_BOLD("JETSTREAM ")
#define _JST_LOG_SEPR          _JST_LOG_BOLD("| ")
#define _JST_LOG_TRACE         _JST_LOG_BOLD("[TRACE] ")
#define _JST_LOG_DEBUG         _JST_LOG_BOLD("[DEBUG] ")
#define _JST_LOG_WARN          _JST_LOG_BOLD("[WARN]  ")
#define _JST_LOG_INFO          _JST_LOG_BOLD("[INFO]  ")
#define _JST_LOG_ERROR         _JST_LOG_BOLD("[ERROR] ")
#define _JST_LOG_FATAL         _JST_LOG_BOLD("[FATAL] ")

#ifndef JST_TRACE
#ifndef NDEBUG
#define JST_TRACE(...) if (getenv("TRACE"))           { \
                       _JST_LOG_SINK                 << \
                       _JST_LOG_NAME                 << \
                       _JST_LOG_TRACE                << \
                       _JST_LOG_SEPR                 << \
                       _JST_LOG_DEFAULT(__VA_ARGS__) << \
                       _JST_LOG_ENDL                  };
#else
#define JST_TRACE(...)
#endif
#endif

#ifndef JST_DEBUG
#ifndef NDEBUG
#define JST_DEBUG(...) _JST_LOG_SINK                << \
                       _JST_LOG_NAME                << \
                       _JST_LOG_DEBUG               << \
                       _JST_LOG_SEPR                << \
                       _JST_LOG_ORANGE(__VA_ARGS__) << \
                       _JST_LOG_ENDL;
#else
#define JST_DEBUG(...)
#endif
#endif

#ifndef JST_WARN
#define JST_WARN(...) _JST_LOG_SINK                << \
                      _JST_LOG_NAME                << \
                      _JST_LOG_WARN                << \
                      _JST_LOG_SEPR                << \
                      _JST_LOG_YELLOW(__VA_ARGS__) << \
                      _JST_LOG_ENDL;
#else
#define JST_WARN(...)
#endif

#ifndef JST_INFO
#define JST_INFO(...) _JST_LOG_SINK              << \
                      _JST_LOG_NAME              << \
                      _JST_LOG_INFO              << \
                      _JST_LOG_SEPR              << \
                      _JST_LOG_CYAN(__VA_ARGS__) << \
                      _JST_LOG_ENDL;
#else
#define JST_INFO(...)
#endif

#ifndef JST_ERROR
#define JST_ERROR(...) _JST_LOG_SINK             << \
                       _JST_LOG_NAME             << \
                       _JST_LOG_ERROR            << \
                       _JST_LOG_SEPR             << \
                       _JST_LOG_RED(__VA_ARGS__) << \
                       _JST_LOG_ENDL;
#else
#define JST_ERROR(...)
#endif

#ifndef JST_FATAL
#define JST_FATAL(...) _JST_LOG_SINK                 << \
                       _JST_LOG_NAME                 << \
                       _JST_LOG_FATAL                << \
                       _JST_LOG_SEPR                 << \
                       _JST_LOG_MAGENTA(__VA_ARGS__) << \
                       _JST_LOG_ENDL;
#else
#define JST_FATAL(...)
#endif

#endif
