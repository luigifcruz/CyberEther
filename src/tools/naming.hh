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

}  // namespace Jetstream::Naming

#endif  // JETSTREAM_TOOLS_NAMING_HH
