#include "jetstream/parser.hh"

namespace Jetstream {

bool ParserMap::contains(const std::string& key) const {
    return index.contains(key);
}

std::any& ParserMap::at(const std::string& key) {
    return entries.at(index.at(key)).value;
}

const std::any& ParserMap::at(const std::string& key) const {
    return entries.at(index.at(key)).value;
}

std::any& ParserMap::operator[](const std::string& key) {
    if (const auto it = index.find(key); it != index.end()) {
        return entries.at(it->second).value;
    }

    const auto position = entries.size();
    entries.push_back({key, std::any{}});
    index[key] = position;
    return entries.back().value;
}

void ParserMap::erase(const std::string& key) {
    const auto it = index.find(key);
    if (it == index.end()) {
        return;
    }

    const auto position = it->second;
    entries.erase(entries.begin() + static_cast<std::vector<Entry>::difference_type>(position));
    index.erase(it);

    for (size_type i = position; i < entries.size(); ++i) {
        index[entries[i].key] = i;
    }
}

void ParserMap::clear() {
    entries.clear();
    index.clear();
}

void ParserMap::reserve(const size_type count) {
    entries.reserve(count);
    index.reserve(count);
}

}  // namespace Jetstream
