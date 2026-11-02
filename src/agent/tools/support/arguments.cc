#include "base.hh"

#include <algorithm>

namespace Jetstream::Agent::Detail {

U64 Arguments::Clamp(const Json& args, const std::string& key, I64 fallback, I64 low, I64 high) {
    I64 requested = fallback;
    if (args.contains(key) && args[key].is_number_integer()) {
        requested = args[key].get<I64>();
    }

    return static_cast<U64>(std::clamp(requested, low, high));
}

}  // namespace Jetstream::Agent::Detail
