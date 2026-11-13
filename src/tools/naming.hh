#ifndef JETSTREAM_TOOLS_NAMING_HH
#define JETSTREAM_TOOLS_NAMING_HH

#include "jetstream/types.hh"

#include <string>

namespace Jetstream::Naming {

template<typename Taken>
inline std::string UniqueName(const std::string& base, const Taken& taken) {
    std::string name = base;
    for (U64 suffix = 1; taken(name); ++suffix) {
        name = base + "_" + std::to_string(suffix);
    }
    return name;
}

template<typename Taken>
inline std::string NextName(const std::string& name, const Taken& taken) {
    if (!taken(name)) {
        return name;
    }

    std::string stem = name;
    U64 next = 1;
    const auto split = name.find_last_of('_');
    const auto digits = split == std::string::npos ? 0 : name.size() - split - 1;
    if (split != std::string::npos && split > 0 && digits > 0 && digits <= 18 &&
        name.find_first_not_of("0123456789", split + 1) == std::string::npos) {
        stem = name.substr(0, split);
        next = std::stoull(name.substr(split + 1)) + 1;
    }

    for (;; ++next) {
        std::string candidate = stem + "_" + std::to_string(next);
        if (!taken(candidate)) {
            return candidate;
        }
    }
}

}  // namespace Jetstream::Naming

#endif  // JETSTREAM_TOOLS_NAMING_HH
