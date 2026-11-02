#include "base.hh"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <locale>
#include <random>
#include <sstream>
#include <thread>

namespace Jetstream::Agent::Detail {

namespace {

using Kind = Provider::Error::Kind;

}  // namespace

std::optional<std::chrono::milliseconds> Retry::After(std::string_view value,
                                                      std::chrono::system_clock::time_point now) {
    using namespace std::chrono;

    const auto first = value.find_first_not_of(" \t");
    if (first == std::string_view::npos) {
        return std::nullopt;
    }
    value = value.substr(first, value.find_last_not_of(" \t") - first + 1);

    const bool numeric = std::all_of(value.begin(), value.end(), [](char c) {
        return c >= '0' && c <= '9';
    });
    if (numeric) {
        U64 seconds = 0;
        for (const char c : value) {
            seconds = seconds * 10 + (c - '0');
            if (seconds > MaxAfter.count()) {
                return milliseconds::max();
            }
        }
        return milliseconds(seconds * 1000);
    }

    const auto formats = {
        "%a, %d %b %Y %H:%M:%S GMT",
        "%A, %d-%b-%y %H:%M:%S GMT",
        "%a %b %d %H:%M:%S %Y",
    };
    for (const char* format : formats) {
        std::tm time{};
        std::istringstream input{std::string(value)};
        input.imbue(std::locale::classic());
        input >> std::get_time(&time, format);
        if (input.fail()) {
            continue;
        }

        input >> std::ws;
        if (!input.eof()) {
            continue;
        }

        const year_month_day date{
            year{time.tm_year + 1900},
            month{static_cast<unsigned>(time.tm_mon + 1)},
            day{static_cast<unsigned>(time.tm_mday)},
        };
        const bool validTime = time.tm_hour <= 23 && time.tm_min <= 59 && time.tm_sec <= 60;
        if (!date.ok() || !validTime) {
            continue;
        }

        const auto midnight = duration_cast<seconds>(sys_days(date).time_since_epoch());
        const auto target = midnight + hours(time.tm_hour) + minutes(time.tm_min) + seconds(time.tm_sec);
        const auto wholeNow = floor<seconds>(now);
        const auto delta = target - wholeNow.time_since_epoch();

        if (delta > MaxAfter) {
            return milliseconds::max();
        }
        if (delta <= seconds::zero()) {
            return milliseconds::zero();
        }
        return ceil<milliseconds>(delta - (now - wholeNow));
    }

    return std::nullopt;
}

std::optional<std::chrono::milliseconds> Retry::Delay(const Provider::Error& failure, U32 attempt, U32 maxRetries,
                                                      std::chrono::system_clock::time_point now) {
    using Kind = Provider::Error::Kind;

    const bool retryable = failure.kind == Kind::RateLimit ||
                           failure.kind == Kind::Server ||
                           failure.kind == Kind::Timeout ||
                           failure.kind == Kind::Connection;
    if (attempt >= maxRetries || !retryable) {
        return std::nullopt;
    }

    if (const auto delay = After(failure.retryAfter, now)) {
        if (*delay > MaxAfter) {
            return std::nullopt;
        }
        return delay;
    }

    const U64 backoff = std::min<U64>(1000ULL << std::min<U32>(attempt, 5), 30000);
    static thread_local std::mt19937 random(std::random_device{}());
    std::uniform_int_distribution<U64> jitter(backoff * 3 / 4, backoff);
    return std::chrono::milliseconds(jitter(random));
}

bool Retry::Wait(std::chrono::milliseconds delay, const Cancellation& cancelled) {
    const auto deadline = std::chrono::steady_clock::now() + delay;
    const auto step = std::chrono::steady_clock::duration(std::chrono::milliseconds(20));

    while (!cancelled || !cancelled()) {
        const auto remaining = deadline - std::chrono::steady_clock::now();
        if (remaining <= remaining.zero()) {
            return true;
        }
        std::this_thread::sleep_for(std::min(remaining, step));
    }

    return false;
}

}  // namespace Jetstream::Agent::Detail
