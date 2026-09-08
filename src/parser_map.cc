#include "jetstream/parser.hh"

namespace Jetstream {

Parser::Sequence Parser::MakeSequence(std::initializer_list<std::any> values) {
    Sequence sequence(values);
    for (auto& value : sequence) {
        if (const auto* text = std::any_cast<const char*>(&value)) {
            value = std::string(*text ? *text : "");
        }
    }
    return sequence;
}

ParserMap::ParserMap(std::initializer_list<Entry> values) {
    reserve(values.size());
    for (const auto& entry : values) {
        if (const auto* text = std::any_cast<const char*>(&entry.value)) {
            (*this)[entry.key] = std::string(*text ? *text : "");
        } else {
            (*this)[entry.key] = entry.value;
        }
    }
}

bool ParserMap::operator==(const ParserMap& other) const {
    if (size() != other.size()) {
        return false;
    }
    for (const auto& [key, value] : entries) {
        if (!other.contains(key) || !Parser::Equal(value, other.at(key))) {
            return false;
        }
    }
    return true;
}

bool Parser::Equal(const std::any& lhs, const std::any& rhs) {
    if (lhs.type() != rhs.type()) {
        return false;
    }
    if (!lhs.has_value()) {
        return true;
    }
    if (const auto* map = std::any_cast<Map>(&lhs)) {
        return *map == std::any_cast<const Map&>(rhs);
    }
    if (const auto* sequence = std::any_cast<Sequence>(&lhs)) {
        const auto& other = std::any_cast<const Sequence&>(rhs);
        return sequence->size() == other.size() &&
               std::equal(sequence->begin(), sequence->end(), other.begin(), Equal);
    }
    bool equal = false;
    const auto compare = [&]<typename T>() {
        if (const auto* value = std::any_cast<T>(&lhs)) {
            equal = *value == std::any_cast<const T&>(rhs);
            return true;
        }
        return false;
    };
    if (compare.template operator()<std::string>() ||
           compare.template operator()<bool>() ||
           compare.template operator()<I8>() || compare.template operator()<U8>() ||
           compare.template operator()<I16>() || compare.template operator()<U16>() ||
           compare.template operator()<I32>() || compare.template operator()<U32>() ||
           compare.template operator()<I64>() || compare.template operator()<U64>() ||
           compare.template operator()<F32>() || compare.template operator()<F64>() ||
           compare.template operator()<CF32>() || compare.template operator()<CF64>() ||
           compare.template operator()<DeviceType>() || compare.template operator()<RuntimeType>() ||
           compare.template operator()<SchedulerType>() ||
           compare.template operator()<Range<F32>>() ||
           compare.template operator()<Extent2D<U64>>() || compare.template operator()<Extent2D<F32>>() ||
           compare.template operator()<std::vector<std::string>>() ||
           compare.template operator()<std::vector<std::vector<std::string>>>() ||
           compare.template operator()<std::vector<U64>>() ||
           compare.template operator()<std::vector<F32>>() || compare.template operator()<std::vector<F64>>() ||
           compare.template operator()<std::vector<CF32>>() || compare.template operator()<std::vector<CF64>>()) {
        return equal;
    }
    JST_ERROR("[PARSER] Equality is not supported for payload type '{}'.", lhs.type().name());
    return false;
}

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
