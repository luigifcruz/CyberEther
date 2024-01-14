#ifndef JETSTREAM_LOGGER_HH
#define JETSTREAM_LOGGER_HH

#include <iostream>
#include <string>
#include <mutex>

#ifndef JST_FMT_INCLUDED
#define JST_FMT_INCLUDED

//
// Create a namespace alias for fmt.
// This is done to avoid conflicts with other libraries that use fmt.
// I really don't like this, but it's the least janky way I could think.
// TODO: Replace this with <format> when it is generally available.
//

#ifdef FMT_BEGIN_NAMESPACE
#undef FMT_BEGIN_NAMESPACE
#undef FMT_END_NAMESPACE
#endif

#ifdef FMT_USE_INT128
#undef FMT_USE_INT128
#endif
#define FMT_USE_INT128 0

#define FMT_BEGIN_NAMESPACE \
    namespace jst {         \
    namespace fmt {         \
    inline namespace v10 {
#define FMT_END_NAMESPACE   \
    }                       \
    }                       \
    }

#undef FMT_ARGS_H_
#undef FMT_CHRONO_H_
#undef FMT_COLOR_H_
#undef FMT_COMPILE_H_
#undef FMT_CORE_H_
#undef FMT_FORMAT_INL_H_
#undef FMT_FORMAT_H_
#undef FMT_OS_H_
#undef FMT_OSTREAM_H_
#undef FMT_PRINTF_H_
#undef FMT_RANGES_H_
#undef FMT_STD_H_
#undef FMT_XCHAR_H_

#define FMT_HEADER_ONLY
#include "jetstream/tools/fmt/format.h"
#include "jetstream/tools/fmt/color.h"
#include "jetstream/tools/fmt/ostream.h"
#include "jetstream/tools/fmt/ranges.h"

#undef FMT_BEGIN_NAMESPACE
#undef FMT_END_NAMESPACE

#undef FMT_ARGS_H_
#undef FMT_CHRONO_H_
#undef FMT_COLOR_H_
#undef FMT_COMPILE_H_
#undef FMT_CORE_H_
#undef FMT_FORMAT_INL_H_
#undef FMT_FORMAT_H_
#undef FMT_OS_H_
#undef FMT_OSTREAM_H_
#undef FMT_PRINTF_H_
#undef FMT_RANGES_H_
#undef FMT_STD_H_
#undef FMT_XCHAR_H_

#endif  // JST_FMT_INCLUDED

#include "jetstream/config.hh"

std::string& JST_LOG_LAST_WARNING();
std::string& JST_LOG_LAST_ERROR();
std::string& JST_LOG_LAST_FATAL();

void JST_LOG_SET_DEBUG_LEVEL(int level);

#ifdef JST_DEBUG_MODE
#define JST_LOG_DEBUG_DEFAULT_LEVEL 3
#else
#define JST_LOG_DEBUG_DEFAULT_LEVEL 2
#endif

std::mutex& _JST_LOG_MUTEX();
int& _JST_LOG_DEBUG_LEVEL();

#define _JST_LOG_SINK          std::cout
#define _JST_LOG_ENDL          std::endl;
#define _JST_LOG_FORMAT        jst::fmt::format

#if defined(JST_OS_IOS) or defined(JST_OS_BROWSER)
#define _JST_LOG_TAINT(first, ...)   _JST_LOG_FORMAT(__VA_ARGS__)
#else
#define _JST_LOG_TAINT(...)          _JST_LOG_FORMAT(__VA_ARGS__)
#endif

#define _JST_LOG_DEFAULT(...)  _JST_LOG_FORMAT(__VA_ARGS__)
#define _JST_LOG_BOLD(...)     _JST_LOG_TAINT(jst::fmt::emphasis::bold, __VA_ARGS__)
#define _JST_LOG_ORANGE(...)   _JST_LOG_TAINT(jst::fmt::fg(jst::fmt::color::orange), __VA_ARGS__)
#define _JST_LOG_YELLOW(...)   _JST_LOG_TAINT(jst::fmt::fg(jst::fmt::color::yellow), __VA_ARGS__)
#define _JST_LOG_CYAN(...)     _JST_LOG_TAINT(jst::fmt::fg(jst::fmt::color::aqua), __VA_ARGS__)
#define _JST_LOG_RED(...)      _JST_LOG_TAINT(jst::fmt::fg(jst::fmt::color::red), __VA_ARGS__)
#define _JST_LOG_MAGENTA(...)  _JST_LOG_TAINT(jst::fmt::fg(jst::fmt::color::fuchsia), __VA_ARGS__)

#define _JST_LOG_NAME          _JST_LOG_BOLD("JETSTREAM ")
#define _JST_LOG_SEPR          _JST_LOG_BOLD("| ")
#define _JST_LOG_TRACE         _JST_LOG_BOLD("[TRACE] ")
#define _JST_LOG_DEBUG         _JST_LOG_BOLD("[DEBUG] ")
#define _JST_LOG_WARN          _JST_LOG_BOLD("[WARN]  ")
#define _JST_LOG_INFO          _JST_LOG_BOLD("[INFO]  ")
#define _JST_LOG_ERROR         _JST_LOG_BOLD("[ERROR] ")
#define _JST_LOG_FATAL         _JST_LOG_BOLD("[FATAL] ")

#ifndef JST_TRACE
#ifdef JST_DEBUG_MODE
#define JST_TRACE(...) if (_JST_LOG_DEBUG_LEVEL() >= 4) { \
                       std::lock_guard<std::mutex> lock(_JST_LOG_MUTEX()); \
                       _JST_LOG_SINK                 << \
                       _JST_LOG_NAME                 << \
                       _JST_LOG_TRACE                << \
                       _JST_LOG_SEPR                 << \
                       _JST_LOG_DEFAULT(__VA_ARGS__) << \
                       _JST_LOG_ENDL; }
#else
#define JST_TRACE(...)
#endif
#endif

#ifndef JST_DEBUG
#define JST_DEBUG(...) if (_JST_LOG_DEBUG_LEVEL() >= 3) { \
                       std::lock_guard<std::mutex> lock(_JST_LOG_MUTEX()); \
                       _JST_LOG_SINK                << \
                       _JST_LOG_NAME                << \
                       _JST_LOG_DEBUG               << \
                       _JST_LOG_SEPR                << \
                       _JST_LOG_ORANGE(__VA_ARGS__) << \
                       _JST_LOG_ENDL; }
#else
#define JST_DEBUG(...)
#endif

#ifndef JST_INFO
#define JST_INFO(...) if (_JST_LOG_DEBUG_LEVEL() >= 2) { \
                      std::lock_guard<std::mutex> lock(_JST_LOG_MUTEX()); \
                      _JST_LOG_SINK              << \
                      _JST_LOG_NAME              << \
                      _JST_LOG_INFO              << \
                      _JST_LOG_SEPR              << \
                      _JST_LOG_CYAN(__VA_ARGS__) << \
                      _JST_LOG_ENDL; }
#else
#define JST_INFO(...)
#endif

#ifndef JST_WARN
#define JST_WARN(...) if (_JST_LOG_DEBUG_LEVEL() >= 1) { \
                      JST_LOG_LAST_WARNING() = _JST_LOG_DEFAULT(__VA_ARGS__); \
                      std::lock_guard<std::mutex> lock(_JST_LOG_MUTEX()); \
                      _JST_LOG_SINK                << \
                      _JST_LOG_NAME                << \
                      _JST_LOG_WARN                << \
                      _JST_LOG_SEPR                << \
                      _JST_LOG_YELLOW(__VA_ARGS__) << \
                      _JST_LOG_ENDL; }
#else
#define JST_WARN(...)
#endif

#ifndef JST_ERROR
#define JST_ERROR(...) if (_JST_LOG_DEBUG_LEVEL() >= 0) { \
                       JST_LOG_LAST_ERROR() = _JST_LOG_DEFAULT(__VA_ARGS__); \
                       std::lock_guard<std::mutex> lock(_JST_LOG_MUTEX()); \
                       _JST_LOG_SINK             << \
                       _JST_LOG_NAME             << \
                       _JST_LOG_ERROR            << \
                       _JST_LOG_SEPR             << \
                       _JST_LOG_RED(__VA_ARGS__) << \
                       _JST_LOG_ENDL; }
#else
#define JST_ERROR(...)
#endif

#ifndef JST_FATAL
#define JST_FATAL(...) if (_JST_LOG_DEBUG_LEVEL() >= 0) { \
                       JST_LOG_LAST_FATAL() = _JST_LOG_DEFAULT(__VA_ARGS__); \
                       std::lock_guard<std::mutex> lock(_JST_LOG_MUTEX()); \
                       _JST_LOG_SINK                 << \
                       _JST_LOG_NAME                 << \
                       _JST_LOG_FATAL                << \
                       _JST_LOG_SEPR                 << \
                       _JST_LOG_MAGENTA(__VA_ARGS__) << \
                       _JST_LOG_ENDL; }
#else
#define JST_FATAL(...)
#endif

#endif
