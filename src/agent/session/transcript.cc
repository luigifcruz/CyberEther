#include "transcript.hh"

#include "jetstream/config.hh"
#include "jetstream/logger.hh"
#include "jetstream/platform.hh"
#include "tools/base64.hh"
#include "tools/text.hh"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <exception>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#if defined(JST_OS_WINDOWS)
#include <io.h>
#include <share.h>
#else
#include <unistd.h>
#endif

namespace Jetstream::Agent::Detail {

namespace {

using Clock = std::chrono::system_clock;
using Kind = Session::Segment::Kind;

constexpr U64 kResultBytes = 16 * 1024;
constexpr U64 kNameBytes = 64;
constexpr U64 kTitleBytes = 200;
constexpr U64 kImageProbeBytes = 64 * 1024;
constexpr U64 kNameAttempts = 10000;
constexpr U64 kCheckpointBytes = 4 * 1024;
constexpr auto kCheckpointInterval = std::chrono::seconds(2);
constexpr U64 kStreamFence = 4;

std::tm Utc(const Clock::time_point time) {
    const std::time_t seconds = Clock::to_time_t(time);
    std::tm utc{};
#if defined(JST_OS_WINDOWS)
    gmtime_s(&utc, &seconds);
#else
    gmtime_r(&seconds, &utc);
#endif
    return utc;
}

std::string Stamp(const Clock::time_point time) {
    const auto utc = Utc(time);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count();
    return jst::fmt::format("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z", utc.tm_year + 1900, utc.tm_mon + 1,
                            utc.tm_mday, utc.tm_hour, utc.tm_min, utc.tm_sec, ((elapsed % 1000) + 1000) % 1000);
}

std::string FileStamp(const Clock::time_point time) {
    const auto utc = Utc(time);
    return jst::fmt::format("{:04}{:02}{:02}-{:02}{:02}{:02}", utc.tm_year + 1900, utc.tm_mon + 1, utc.tm_mday,
                            utc.tm_hour, utc.tm_min, utc.tm_sec);
}

std::string FileToken(const std::string_view value) {
    std::string token;
    for (const char ch : value.substr(0, kNameBytes)) {
        const bool keep = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') ||
                          ch == '.' || ch == '_' || ch == '-';
        token += keep ? ch : '_';
    }
    return token.empty() ? "unknown" : token;
}

// Headings and list items hold a single line, whatever the model or a tool put in a name.
std::string Line(const std::string_view value, const U64 limit = std::string_view::npos) {
    std::string line(Text::Utf8Prefix(value, limit));
    for (char& ch : line) {
        if (static_cast<unsigned char>(ch) < 0x20 || ch == 0x7f) {
            ch = ' ';
        }
    }
    return line;
}

std::string Named(const std::string& value, const char* fallback) {
    return Text::IsBlank(value) ? std::string(fallback) : Line(value);
}

U64 Backticks(const std::string_view content) {
    U64 longest = 0;
    U64 run = 0;
    for (const char ch : content) {
        run = ch == '`' ? run + 1 : 0;
        longest = std::max(longest, run);
    }
    return longest;
}

// A fence one backtick longer than any run inside the content cannot be closed by it.
void Block(std::string& out, const std::string_view language, const std::string_view content) {
    const std::string fence(std::max<U64>(3, Backticks(content) + 1), '`');

    out += fence;
    out += language;
    out += '\n';
    out += content;
    if (!content.empty() && content.back() != '\n') {
        out += '\n';
    }
    out += fence;
    out += '\n';
}

U64 BigEndian(const std::vector<U8>& bytes, const U64 offset, const U64 count) {
    U64 value = 0;
    for (U64 i = 0; i < count; ++i) {
        value = (value << 8) | bytes[offset + i];
    }
    return value;
}

std::optional<std::pair<U64, U64>> PngSize(const std::vector<U8>& bytes) {
    static constexpr U8 kSignature[] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    static constexpr U8 kHeader[] = {'I', 'H', 'D', 'R'};
    if (bytes.size() < 24 || !std::equal(std::begin(kSignature), std::end(kSignature), bytes.begin()) ||
        !std::equal(std::begin(kHeader), std::end(kHeader), bytes.begin() + 12)) {
        return std::nullopt;
    }
    return std::pair{BigEndian(bytes, 16, 4), BigEndian(bytes, 20, 4)};
}

std::optional<std::pair<U64, U64>> JpegSize(const std::vector<U8>& bytes) {
    if (bytes.size() < 4 || bytes[0] != 0xff || bytes[1] != 0xd8) {
        return std::nullopt;
    }

    U64 offset = 2;
    while (offset + 4 <= bytes.size()) {
        if (bytes[offset] != 0xff) {
            return std::nullopt;
        }

        const U8 marker = bytes[offset + 1];
        if (marker == 0xff) {
            ++offset;
            continue;
        }
        if (marker == 0x01 || (marker >= 0xd0 && marker <= 0xd8)) {
            offset += 2;
            continue;
        }
        if (marker == 0xd9 || marker == 0xda) {
            return std::nullopt;
        }

        const bool frame = marker >= 0xc0 && marker <= 0xcf && marker != 0xc4 && marker != 0xc8 && marker != 0xcc;
        if (frame) {
            if (offset + 9 > bytes.size()) {
                return std::nullopt;
            }
            return std::pair{BigEndian(bytes, offset + 7, 2), BigEndian(bytes, offset + 5, 2)};
        }

        offset += 2 + BigEndian(bytes, offset + 2, 2);
    }
    return std::nullopt;
}

// Only the dimensions are ever written; the pixels and any URL stay out of the file.
std::string ImageNote(const std::string& image) {
    static constexpr std::string_view kScheme = "data:";
    static constexpr std::string_view kEncoding = ";base64";

    const auto comma = image.find(',');
    if (!image.starts_with(kScheme) || comma == std::string::npos ||
        !std::string_view(image).substr(0, comma).ends_with(kEncoding)) {
        return "[image]";
    }

    U64 length = std::min<U64>(image.size() - comma - 1, kImageProbeBytes);
    length -= length % 4;

    const auto bytes = Base64::Decode(image.substr(comma + 1, length));
    auto size = PngSize(bytes);
    if (!size) {
        size = JpegSize(bytes);
    }
    if (!size || size->first == 0 || size->second == 0) {
        return "[image]";
    }
    return jst::fmt::format("[image {}x{}]", size->first, size->second);
}

std::string ImageNotes(const std::span<const std::string> images) {
    std::string notes;
    for (const auto& image : images) {
        if (!notes.empty()) {
            notes += ", ";
        }
        notes += ImageNote(image);
    }
    return notes;
}

const char* Title(const Kind kind) {
    switch (kind) {
        case Kind::Reasoning:
            return "Assistant reasoning";
        case Kind::Summary:
            return "Assistant reasoning summary";
        case Kind::Text:
            return "Assistant";
        case Kind::Error:
            return "Error";
        case Kind::Status:
            return "Status";
        case Kind::Note:
        case Kind::ToolCall:
        case Kind::ToolResult:
            break;
    }
    return "Note";
}

bool SameOrigin(const Transcript::Origin& a, const Transcript::Origin& b) {
    return a.provider == b.provider && a.model == b.model && a.reasoningEffort == b.reasoningEffort;
}

std::string Describe(const Transcript::Origin& origin) {
    return "- Provider: " + Named(origin.provider, "unknown") + "\n" +
           "- Model: " + Named(origin.model, "unknown") + "\n" +
           "- Reasoning effort: " + Named(origin.reasoningEffort, "default") + "\n";
}

void Replace(std::string& text, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return;
    }
    for (auto at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size())) {
        text.replace(at, from.size(), to);
    }
}

// Creates the file of one conversation. The creation is exclusive, so a name that another
// chat or another running CyberEther took in the same second is never shared, and the file
// is private to the user from its first byte.
std::FILE* Create(const std::filesystem::path& path, int& error) {
    error = 0;
#if defined(JST_OS_WINDOWS)
    int descriptor = -1;
    error = _wsopen_s(&descriptor, path.c_str(),
                      _O_WRONLY | _O_CREAT | _O_EXCL | _O_APPEND | _O_BINARY | _O_NOINHERIT, _SH_DENYWR,
                      _S_IREAD | _S_IWRITE);
    if (error != 0) {
        return nullptr;
    }

    std::FILE* file = _fdopen(descriptor, "ab");
    if (!file) {
        error = errno;
        _close(descriptor);
    }
#else
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_CLOEXEC,
                                  S_IRUSR | S_IWUSR);
    if (descriptor < 0) {
        error = errno;
        return nullptr;
    }

    std::FILE* file = ::fdopen(descriptor, "ab");
    if (!file) {
        error = errno;
        ::close(descriptor);
    }
#endif
    return file;
}

}  // namespace

struct Transcript::Impl {
    std::mutex mutex;
    std::string folder;
    std::FILE* file = nullptr;
    bool opened = false;
    bool failed = false;
    Origin origin;
    U64 rounds = 0;

    // The segment being streamed: `buffered` is the part that has not reached the disk, and
    // `fence` is the length of the fence a checkpoint left open (zero when none is open).
    std::optional<Kind> pending;
    std::string buffered;
    Clock::time_point began;
    Clock::time_point saved;
    U64 fence = 0;
    bool lineEnded = true;
    bool continued = false;

    ~Impl() {
        if (file) {
            std::fclose(file);
        }
    }

    template<typename Action>
    void guarded(Action&& action) {
        std::lock_guard lock(mutex);
        if (failed) {
            return;
        }

        try {
            action();
        } catch (const std::exception& exception) {
            disable(exception.what());
        } catch (...) {
            disable("unknown error");
        }
    }

    void disable(const std::string& reason) noexcept {
        const bool first = !failed;
        failed = true;
        pending.reset();
        buffered = {};
        fence = 0;
        continued = false;

        if (file) {
            std::fclose(file);
            file = nullptr;
        }

        try {
            if (first) {
                JST_WARN("[AGENT] Chat transcript disabled for this conversation (folder '{}'): {}", folder, reason);
            }
        } catch (...) {
        }
    }

    bool open(const Clock::time_point time) {
        if (opened) {
            return !failed;
        }
        opened = true;

        if (folder.empty()) {
            disable("no transcript folder is configured");
            return false;
        }

        const auto directory = Platform::PathFromUtf8(folder);
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (error || !std::filesystem::is_directory(directory, error)) {
            disable(error ? error.message() : "the path is not a folder");
            return false;
        }

        const auto stem = FileStamp(time) + "-" + FileToken(origin.provider) + "-" + FileToken(origin.model);
        int status = EEXIST;
        for (U64 attempt = 1; attempt <= kNameAttempts && !file && status == EEXIST; ++attempt) {
            const auto name = attempt == 1 ? stem + ".md" : stem + "-" + std::to_string(attempt) + ".md";
            file = Create(directory / Platform::PathFromUtf8(name), status);
        }
        if (!file) {
            disable("the transcript file could not be created: " + std::generic_category().message(status));
            return false;
        }

        return put("# CyberEther agent transcript\n\n"
                   "- Started: " + Stamp(time) + " (all times are UTC)\n" +
                   Describe(origin) +
                   "- App version: " JETSTREAM_VERSION_STR "\n");
    }

    bool put(const std::string_view text) {
        if (!file) {
            return false;
        }
        if (std::fwrite(text.data(), 1, text.size(), file) != text.size() || std::fflush(file) != 0) {
            disable("writing to the transcript file failed");
            return false;
        }
        return true;
    }

    void entry(const Clock::time_point time, const std::string& title, const std::string& body) {
        if (!open(time)) {
            return;
        }
        put("\n## " + Stamp(time) + " " + title + "\n\n" + body);
    }

    // Provider failures name a custom endpoint by its base URL, which can carry credentials.
    // Errors and status lines get the provider's name instead, and no URL keeps a password.
    std::string scrub(std::string text) const {
        const auto name = Named(origin.provider, "the provider");
        Replace(text, origin.host, name);
        Replace(text, Text::WithoutUrlCredentials(origin.host), name);
        return Text::WithoutUrlCredentials(std::move(text));
    }

    // A long segment reaches the disk while it is still streaming, so a crash or a kill loses
    // a couple of seconds of it at most. The fence stays open and later chunks are appended.
    void checkpoint(const Clock::time_point now) {
        const bool due = buffered.size() >= kCheckpointBytes || now - saved >= kCheckpointInterval || now < saved;
        if (!due) {
            return;
        }

        // A run of backticks at the end may still be growing, so it waits for the next chunk.
        const auto last = buffered.find_last_not_of('`');
        if (last == std::string::npos) {
            return;
        }

        saved = now;
        if (extend(std::string_view(buffered).substr(0, last + 1), now)) {
            buffered.erase(0, last + 1);
        }
    }

    // Appends to the open entry of the pending segment, starting one when none is open.
    bool extend(const std::string_view piece, const Clock::time_point now) {
        const U64 longest = Backticks(piece);
        if (fence != 0 && longest >= fence) {
            // The open fence could be closed by this piece, so the segment moves to a new entry.
            if (!close()) {
                return false;
            }
            began = now;
        }

        if (fence != 0) {
            if (!put(piece)) {
                return false;
            }
        } else {
            if (!open(began)) {
                return false;
            }

            const U64 length = std::max(kStreamFence, longest + 1);
            std::string head = "\n## " + Stamp(began) + " " + Title(*pending) + (continued ? " (continued)" : "") +
                               "\n\n" + std::string(length, '`') + "text\n";
            head += piece;
            if (!put(head)) {
                return false;
            }
            fence = length;
        }

        lineEnded = piece.back() == '\n';
        return true;
    }

    bool close() {
        std::string tail = lineEnded ? "" : "\n";
        tail.append(fence, '`');
        tail += '\n';

        fence = 0;
        continued = true;
        return put(tail);
    }

    void settle() {
        if (!pending) {
            return;
        }

        const auto kind = *pending;
        if (fence == 0) {
            const bool diagnostic = kind == Kind::Error || kind == Kind::Status;

            std::string body;
            Block(body, "text", diagnostic ? scrub(std::move(buffered)) : buffered);
            entry(began, Title(kind), body);
        } else if (buffered.empty() || extend(buffered, Clock::now())) {
            close();
        }

        pending.reset();
        buffered = {};
        fence = 0;
        continued = false;
    }
};

Transcript::Transcript(std::string folder) : impl(std::make_unique<Impl>()) {
    impl->folder = std::move(folder);
}

Transcript::~Transcript() {
    flush();
}

void Transcript::user(const Origin& origin, const std::string& text, const std::span<const std::string> images) {
    impl->guarded([&] {
        const auto now = Clock::now();
        impl->settle();

        const bool changed = impl->opened && !SameOrigin(impl->origin, origin);
        impl->origin = origin;
        if (changed) {
            impl->entry(now, "Model changed", Describe(origin));
        }

        std::string body;
        if (!images.empty()) {
            body += "- Images: " + ImageNotes(images) + "\n\n";
        }
        Block(body, "text", text);
        impl->entry(now, "User", body);
    });
}

void Transcript::stream(const Kind kind, const std::string& chunk) {
    if (chunk.empty()) {
        return;
    }

    impl->guarded([&] {
        const auto now = Clock::now();
        if (impl->pending && *impl->pending != kind) {
            impl->settle();
        }
        if (!impl->pending) {
            impl->pending = kind;
            impl->began = now;
            impl->saved = now;
        }
        impl->buffered += chunk;

        // Status lines and errors arrive whole, so they reach the disk at once.
        if (kind == Kind::Status || kind == Kind::Error) {
            impl->settle();
        } else {
            impl->checkpoint(now);
        }
    });
}

void Transcript::call(const Provider::FunctionCall& call, const bool run) {
    impl->guarded([&] {
        const auto now = Clock::now();
        impl->settle();

        std::string body = "- Call id: " + Named(call.id, "none") + "\n\n";
        Block(body, "json", call.arguments);
        impl->entry(now, (run ? "Tool call: " : "Tool call (not run): ") + Line(call.name, kTitleBytes), body);
    });
}

void Transcript::result(const Provider::FunctionCall& call, const Tools::Output& output) {
    impl->guarded([&] {
        const auto now = Clock::now();
        impl->settle();

        std::string body = "- Call id: " + Named(call.id, "none") + "\n";
        body += output.error ? "- Error: yes\n" : "- Error: no\n";
        if (!output.images.empty()) {
            body += "- Images: " + ImageNotes(output.images) + "\n";
        }
        body += '\n';

        const auto shown = Text::Utf8Prefix(output.content, kResultBytes);
        Block(body, "text", shown);
        if (shown.size() < output.content.size()) {
            body += jst::fmt::format("\n[truncated, {} bytes total]\n", output.content.size());
        }
        impl->entry(now, "Tool result: " + Line(call.name, kTitleBytes), body);
    });
}

void Transcript::round(const Provider::Response& response, const bool summary) {
    impl->guarded([&] {
        const auto now = Clock::now();
        impl->settle();

        const auto& usage = response.usage;
        const auto& error = response.error;

        std::string body;
        if (usage.inputTokens || usage.outputTokens || usage.cachedTokens) {
            body += jst::fmt::format("- Tokens: prompt {}, completion {}, cached {}", usage.inputTokens,
                                     usage.outputTokens, usage.cachedTokens);
            if (usage.reasoningTokens) {
                body += jst::fmt::format(", reasoning {}", usage.reasoningTokens);
            }
            if (usage.cacheWriteTokens) {
                body += jst::fmt::format(", cache write {}", usage.cacheWriteTokens);
            }
            body += '\n';
        } else {
            body += "- Tokens: not reported\n";
        }
        if (!response.toolCalls.empty()) {
            body += jst::fmt::format("- Tool calls requested: {}\n", response.toolCalls.size());
        }
        if (error) {
            body += "- Error: " + Named(impl->scrub(error.message), "unspecified");
            if (error.status != 0) {
                body += jst::fmt::format(" (HTTP {})", error.status);
            }
            if (!error.code.empty()) {
                body += " [" + Line(impl->scrub(error.code)) + "]";
            }
            if (!error.retryAfter.empty()) {
                body += " retry after " + Line(impl->scrub(error.retryAfter));
            }
            body += '\n';
        }

        auto title = jst::fmt::format("Model round {}", ++impl->rounds);
        if (summary) {
            title += " (context summary)";
        }
        impl->entry(now, title, body);
    });
}

void Transcript::event(const std::string& title, const std::string& text) {
    impl->guarded([&] {
        const auto now = Clock::now();
        impl->settle();

        std::string body;
        Block(body, "text", text);
        impl->entry(now, Line(title, kTitleBytes), body);
    });
}

void Transcript::flush() {
    impl->guarded([&] {
        impl->settle();
    });
}

}  // namespace Jetstream::Agent::Detail
