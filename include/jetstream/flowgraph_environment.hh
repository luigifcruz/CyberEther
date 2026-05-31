#ifndef JETSTREAM_FLOWGRAPH_ENVIRONMENT_HH
#define JETSTREAM_FLOWGRAPH_ENVIRONMENT_HH

#include <any>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "jetstream/flowgraph.hh"
#include "jetstream/logger.hh"
#include "jetstream/parser.hh"

namespace Jetstream {

class JETSTREAM_API Flowgraph::Environment {
 public:
    explicit Environment(const std::shared_ptr<Flowgraph::Impl>& impl);

    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;

    bool has(const std::string& key,
             U64 timestamp = std::numeric_limits<U64>::min()) const;

    template<typename T>
    Result get(const std::string& key,
               T& data,
               U64 timestamp = std::numeric_limits<U64>::min()) const {
        Parser::Map stored;
        JST_CHECK(get(key, stored, timestamp));

        if (stored.empty()) {
            return Result::SUCCESS;
        }

        Parser::Map encoded;
        encoded[key] = stored;
        return Parser::Deserialize(encoded, key, data);
    }

    Result get(const std::string& key,
               Parser::Map& data,
               U64 timestamp = std::numeric_limits<U64>::min()) const;
    Result keys(std::vector<std::string>& keys) const;

    template<typename T>
    bool tryGet(const std::string& key,
                T& data,
                U64 timestamp = std::numeric_limits<U64>::min()) const {
        if (!has(key, timestamp)) {
            return false;
        }

        return get(key, data, timestamp) == Result::SUCCESS;
    }

    template<typename T>
    Result set(const std::string& key,
               const T& data,
               U64 start = std::numeric_limits<U64>::min(),
               U64 end = std::numeric_limits<U64>::max()) {
        Parser::Map encoded;
        JST_CHECK(Parser::Serialize(encoded, key, data));

        if (!encoded.contains(key) || encoded.at(key).type() != typeid(Parser::Map)) {
            JST_ERROR("[FLOWGRAPH] Environment value '{}' must serialize to a map.", key);
            return Result::ERROR;
        }

        return set(key, std::any_cast<const Parser::Map&>(encoded.at(key)), start, end);
    }

    Result set(const std::string& key,
               const Parser::Map& data,
               U64 start = std::numeric_limits<U64>::min(),
               U64 end = std::numeric_limits<U64>::max());
    Result clear(const std::string& key);
    Result clearAll();

 private:
    std::weak_ptr<Flowgraph::Impl> impl;

    friend class Flowgraph;
};

}  // namespace Jetstream

#endif  // JETSTREAM_FLOWGRAPH_ENVIRONMENT_HH
