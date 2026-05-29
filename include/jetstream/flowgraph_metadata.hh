#ifndef JETSTREAM_FLOWGRAPH_METADATA_HH
#define JETSTREAM_FLOWGRAPH_METADATA_HH

#include <any>
#include <memory>
#include <string>

#include "jetstream/flowgraph.hh"
#include "jetstream/logger.hh"
#include "jetstream/parser.hh"

namespace Jetstream {

class JETSTREAM_API Flowgraph::Metadata {
 public:
    explicit Metadata(const std::shared_ptr<Flowgraph::Impl>& impl);

    Metadata(const Metadata&) = delete;
    Metadata& operator=(const Metadata&) = delete;

    bool has(const std::string& key, const std::string& block = {}) const;

    template<typename T>
    Result get(const std::string& key, T& data, const std::string& block = {}) const {
        Parser::Map stored;
        JST_CHECK(get(key, stored, block));

        if (stored.empty()) {
            return Result::SUCCESS;
        }

        Parser::Map encoded;
        encoded[key] = stored;
        return Parser::Deserialize(encoded, key, data);
    }

    Result get(const std::string& key, Parser::Map& data, const std::string& block = {}) const;

    template<typename T>
    bool tryGet(const std::string& key, T& data, const std::string& block = {}) const {
        if (!has(key, block)) {
            return false;
        }

        return get(key, data, block) == Result::SUCCESS;
    }

    template<typename T>
    Result set(const std::string& key, const T& data, const std::string& block = {}) {
        Parser::Map encoded;
        JST_CHECK(Parser::Serialize(encoded, key, data));

        if (!encoded.contains(key) || encoded.at(key).type() != typeid(Parser::Map)) {
            JST_ERROR("[FLOWGRAPH] Metadata '{}' must serialize to a map.", key);
            return Result::ERROR;
        }

        return set(key, std::any_cast<const Parser::Map&>(encoded.at(key)), block);
    }

    Result set(const std::string& key, const Parser::Map& data, const std::string& block = {});
    Result clear(const std::string& key, const std::string& block = {});
    Result clearAll();

 private:
    std::weak_ptr<Flowgraph::Impl> impl;

    friend class Flowgraph;
};

}  // namespace Jetstream

#endif  // JETSTREAM_FLOWGRAPH_METADATA_HH
