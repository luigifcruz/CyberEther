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
    U64 end = std::min<U64>(limit, value.size());
    while (end > 0 && end < value.size() &&
           (static_cast<unsigned char>(value[end]) & 0xc0) == 0x80) {
        --end;
    }
    return value.substr(0, end);
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
