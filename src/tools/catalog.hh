#ifndef JETSTREAM_TOOLS_CATALOG_HH
#define JETSTREAM_TOOLS_CATALOG_HH

#include <jetstream/types.hh>

#include <algorithm>
#include <string>

#include <nlohmann/json.hpp>

namespace Jetstream::Catalog {

inline bool ValidDigest(const std::string& digest) {
    return digest.size() == 64 && std::all_of(digest.begin(), digest.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

inline bool SplitUrl(const std::string& url, std::string& host, std::string& path) {
    const auto scheme = url.find("://");
    if (scheme == std::string::npos) {
        return false;
    }
    const auto slash = url.find('/', scheme + 3);
    host = url.substr(0, slash);
    path = slash == std::string::npos ? "/" : url.substr(slash);
    return host.size() > scheme + 3;
}

inline bool ReadText(const nlohmann::json& item, const char* name, std::string& value) {
    const auto field = item.find(name);
    if (field == item.end() || field->is_null()) {
        return true;
    }
    if (!field->is_string()) {
        return false;
    }
    value = field->get<std::string>();
    return true;
}

inline bool ReadSize(const nlohmann::json& item, U64& value) {
    const auto field = item.find("size");
    if (field == item.end() || field->is_null()) {
        return true;
    }
    if (!field->is_number_unsigned()) {
        return false;
    }
    value = field->get<U64>();
    return true;
}

}  // namespace Jetstream::Catalog

#endif  // JETSTREAM_TOOLS_CATALOG_HH
