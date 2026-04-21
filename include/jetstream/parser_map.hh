#ifndef JETSTREAM_PARSER_MAP_HH
#define JETSTREAM_PARSER_MAP_HH

#include <any>
#include <string>
#include <unordered_map>
#include <vector>

namespace Jetstream {

class ParserMap {
 public:
    struct Entry {
        std::string key;
        std::any value;
    };

    using iterator = std::vector<Entry>::iterator;
    using const_iterator = std::vector<Entry>::const_iterator;
    using size_type = std::vector<Entry>::size_type;

    bool contains(const std::string& key) const;
    std::any& at(const std::string& key);
    const std::any& at(const std::string& key) const;
    std::any& operator[](const std::string& key);
    void erase(const std::string& key);
    void clear();

    bool empty() const {
        return entries.empty();
    }

    size_type size() const {
        return entries.size();
    }

    void reserve(const size_type count);

    iterator begin() {
        return entries.begin();
    }

    iterator end() {
        return entries.end();
    }

    const_iterator begin() const {
        return entries.begin();
    }

    const_iterator end() const {
        return entries.end();
    }

    const_iterator cbegin() const {
        return entries.cbegin();
    }

    const_iterator cend() const {
        return entries.cend();
    }

 private:
    std::vector<Entry> entries;
    std::unordered_map<std::string, size_type> index;
};

}  // namespace Jetstream

#endif  // JETSTREAM_PARSER_MAP_HH
