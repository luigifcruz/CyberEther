#ifndef JETSTREAM_TOOLS_BASE64_HH
#define JETSTREAM_TOOLS_BASE64_HH

#include "jetstream/types.hh"

#include <c4/base64.hpp>

#include <string>
#include <vector>

namespace Jetstream::Base64 {

inline std::string Encode(const U8* data, const U64 size) {
    const c4::cblob input(reinterpret_cast<const c4::cbyte*>(data), size);
    std::string out(c4::base64_encode({}, input), '\0');
    c4::base64_encode(c4::substr(out.data(), out.size()), input);
    return out;
}

inline std::vector<U8> Decode(const std::string& text) {
    const c4::csubstr input(text.data(), text.size());
    if ((input.len & 3u) != 0 || !c4::base64_valid(input)) {
        return {};
    }
    const auto padding = text.find('=');
    if (padding != std::string::npos &&
        (text.size() - padding > 2 || text.find_first_not_of('=', padding) != std::string::npos)) {
        return {};
    }
    std::vector<U8> out(c4::base64_decode(input, {}));
    c4::base64_decode(input, c4::blob(reinterpret_cast<c4::byte*>(out.data()), out.size()));
    return out;
}

}  // namespace Jetstream::Base64

#endif  // JETSTREAM_TOOLS_BASE64_HH
