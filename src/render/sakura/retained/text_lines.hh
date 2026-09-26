#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_LINES_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_LINES_HH

#include <jetstream/types.hh>

#include <string>
#include <vector>

namespace Jetstream::Sakura::Retained {

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

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_LINES_HH
