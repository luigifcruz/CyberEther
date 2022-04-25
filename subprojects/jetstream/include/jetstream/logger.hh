#ifndef JETSTREAM_LOGGER_HH
#define JETSTREAM_LOGGER_HH

#include <iostream>

#include <fmt/format.h>
#include <fmt/color.h>

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define JST_LOG_HEAD_DECR fmt::emphasis::bold
#define JST_LOG_HEAD_NAME fmt::format(JST_LOG_HEAD_DECR, "JETSTREAM ")
#define JST_LOG_HEAD_FILE fmt::format(JST_LOG_HEAD_DECR, "[{}@{}] ", __FILENAME__, __LINE__)
#define JST_LOG_HEAD_SEPR fmt::format(JST_LOG_HEAD_DECR, "| ")
#define JST_LOG_HEAD JST_LOG_HEAD_NAME << JST_LOG_HEAD_FILE << JST_LOG_HEAD_SEPR 

#if !defined(JST_TRACE) || !defined(NDEBUG)
#define JST_TRACE(...) std::cout << JST_LOG_HEAD << fmt::format(fg(fmt::color::white), __VA_ARGS__) << std::endl;
#endif

#if !defined(JST_DEBUG) || !defined(NDEBUG)
#define JST_DEBUG(...) std::cout << JST_LOG_HEAD << fmt::format(fg(fmt::color::orange), __VA_ARGS__) << std::endl;
#endif

#if !defined(JST_WARN) || defined(NDEBUG)
#define JST_WARN(...) std::cout << JST_LOG_HEAD << fmt::format(fg(fmt::color::yellow), __VA_ARGS__) << std::endl;
#endif

#if !defined(JST_INFO) || defined(NDEBUG)
#define JST_INFO(...) std::cout << JST_LOG_HEAD << fmt::format(fg(fmt::color::cyan), __VA_ARGS__) << std::endl;
#endif

#if !defined(JST_ERROR) || defined(NDEBUG)
#define JST_ERROR(...) std::cerr << JST_LOG_HEAD << fmt::format(fg(fmt::color::red), __VA_ARGS__) << std::endl;
#endif

#if !defined(JST_FATAL) || defined(NDEBUG)
#define JST_FATAL(...) std::cerr << JST_LOG_HEAD << fmt::format(fg(fmt::color::magenta), __VA_ARGS__) << std::endl;
#endif

#endif
