#ifndef JETSTREAM_TOOLS_TEXT_HH
#define JETSTREAM_TOOLS_TEXT_HH

#include <jetstream/types.hh>

#include <string>
#include <string_view>
#include <algorithm>
#include <cctype>
#include <vector>

namespace Jetstream::Text {

inline std::string_view Utf8Prefix(std::string_view value, U64 limit) {
    const U64 cut = std::min<U64>(limit, value.size());
    if (cut == value.size()) {
        return value;
    }

    const auto continues = [&](const U64 index) {
        return (static_cast<unsigned char>(value[index]) & 0xc0) == 0x80;
    };

    // A character has at most three continuation bytes. A longer run is not UTF-8, so the
    // text is cut at the limit itself rather than walked back to nothing.
    U64 end = cut;
    while (end > 0 && cut - end < 3 && continues(end)) {
        --end;
    }
    return value.substr(0, continues(end) ? cut : end);
}

// Drops the "user:password@" part of every URL in the text, so an endpoint can be named in
// a log or an error without its credentials.
inline std::string WithoutUrlCredentials(std::string text) {
    constexpr std::string_view scheme = "://";
    constexpr std::string_view authorityEnd = "/?#\\\"<> \t\r\n";

    std::string::size_type position = 0;
    while ((position = text.find(scheme, position)) != std::string::npos) {
        position += scheme.size();

        auto end = text.find_first_of(authorityEnd, position);
        if (end == std::string::npos) {
            end = text.size();
        }
        if (end == position) {
            continue;
        }

        const auto at = text.rfind('@', end - 1);
        if (at != std::string::npos && at >= position) {
            text.erase(position, at + 1 - position);
        }
    }
    return text;
}

inline std::vector<std::string> SplitLines(const std::string& value) {
    std::vector<std::string> lines;
    std::string::size_type start = 0;
    while (true) {
        const auto end = value.find('\n', start);
        if (end == std::string::npos) {
            lines.push_back(value.substr(start));
            break;
        }
        lines.push_back(value.substr(start, end - start));
        start = end + 1;
    }
    return lines;
}

inline std::string JoinLines(const std::vector<std::string>& lines) {
    std::string value;
    for (U64 i = 0; i < lines.size(); ++i) {
        if (i > 0) {
            value += '\n';
        }
        value += lines[i];
    }
    return value;
}

inline std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

inline constexpr const char* kBlankCharacters = " \t\r\n";

inline bool IsBlank(const std::string& text) {
    return text.find_first_not_of(kBlankCharacters) == std::string::npos;
}

inline std::string TrimBlank(const std::string& text) {
    const auto begin = text.find_first_not_of(kBlankCharacters);
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = text.find_last_not_of(kBlankCharacters);
    return text.substr(begin, end - begin + 1);
}

inline void AppendParagraph(std::string& text, const std::string& paragraph) {
    if (paragraph.empty()) {
        return;
    }
    if (!text.empty()) {
        text += "\n\n";
    }
    text += paragraph;
}

}  // namespace Jetstream::Text

#endif  // JETSTREAM_TOOLS_TEXT_HH
