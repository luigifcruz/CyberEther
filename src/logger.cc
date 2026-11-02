// jetstream/logger.cc
#include <atomic>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

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

namespace {

class RecentLog : public std::streambuf {
 public:
    static constexpr std::size_t Capacity = 256;
    static constexpr std::size_t EntryLimit = 4096;

    std::vector<std::string> lines() const {
        return {history.begin(), history.end()};
    }

    void begin(std::ostream& sink) {
        target = &sink;
        reset();
    }

 protected:
    int_type overflow(int_type c) override {
        if (!traits_type::eq_int_type(c, traits_type::eof())) {
            const char ch = traits_type::to_char_type(c);
            xsputn(&ch, 1);
        }
        return traits_type::not_eof(c);
    }

    std::streamsize xsputn(const char* data, std::streamsize size) override {
        target->write(data, size);
        for (std::streamsize i = 0; i < size; ++i) {
            record(data[i]);
        }
        return size;
    }

    int sync() override {
        commit();
        target->flush();
        return 0;
    }

 private:
    enum class Escape { None, Start, Control, String, StringEnd };

    std::ostream* target = &std::cout;
    std::deque<std::string> history;
    std::string line;
    Escape escape = Escape::None;
    bool truncated = false;

    void reset() {
        line.clear();
        escape = Escape::None;
        truncated = false;
    }

    bool skip(char c) {
        const auto byte = static_cast<unsigned char>(c);
        switch (escape) {
            case Escape::None:
                if (c != '\x1b') {
                    return false;
                }
                escape = Escape::Start;
                return true;
            case Escape::Start:
                escape = c == '[' ? Escape::Control : c == ']' ? Escape::String : Escape::None;
                return true;
            case Escape::Control:
                if (byte >= 0x40 && byte <= 0x7E) {
                    escape = Escape::None;
                }
                return true;
            case Escape::String:
                if (c == '\a') {
                    escape = Escape::None;
                } else if (c == '\x1b') {
                    escape = Escape::StringEnd;
                }
                return true;
            case Escape::StringEnd:
                escape = c == '\\' ? Escape::None : Escape::String;
                return true;
        }
        return false;
    }

    void record(char c) {
        if (skip(c)) {
            return;
        }
        if (c == '\r') {
            return;
        }
        if (line.size() >= EntryLimit) {
            truncated = true;
            return;
        }
        line += c;
    }

    static void TrimPartialCharacter(std::string& text) {
        std::size_t lead = text.size();
        while (lead > 0 && (static_cast<unsigned char>(text[lead - 1]) & 0xC0) == 0x80) {
            --lead;
        }
        if (lead == 0) {
            text.clear();
            return;
        }

        const auto byte = static_cast<unsigned char>(text[lead - 1]);
        const std::size_t length = byte < 0x80 ? 1 : byte >= 0xF0 ? 4 : byte >= 0xE0 ? 3 : byte >= 0xC0 ? 2 : 1;
        if (text.size() - (lead - 1) < length) {
            text.erase(lead - 1);
        }
    }

    void commit() {
        while (!line.empty() && line.back() == '\n') {
            line.pop_back();
        }
        if (truncated) {
            TrimPartialCharacter(line);
        }

        constexpr std::string_view prefix = "JETSTREAM ";
        if (line.starts_with(prefix)) {
            line.erase(0, prefix.size());
        }
        if (!line.empty()) {
            history.push_back(std::move(line));
            if (history.size() > Capacity) {
                history.pop_front();
            }
        }
        reset();
    }
};

RecentLog& Recent() {
    static auto* recent = new RecentLog();
    return *recent;
}

}  // namespace

std::ostream& JST_LOG_SINK() {
    return *g_sink.load(std::memory_order_acquire);
}

std::ostream& _JST_LOG_STREAM() {
    static auto* stream = new std::ostream(&Recent());
    auto& sink = JST_LOG_SINK();
    Recent().begin(sink);
    stream->clear();
    stream->exceptions(sink.exceptions());
    return *stream;
}

std::vector<std::string> JST_LOG_RECENT() {
    std::lock_guard<std::mutex> lock(_JST_LOG_MUTEX());
    return Recent().lines();
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
