#ifndef JETSTREAM_FLOWGRAPH_HH
#define JETSTREAM_FLOWGRAPH_HH

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <iostream>
#include <unordered_map>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/parser.hh"
#include "jetstream/block.hh"
#include "jetstream/module.hh"

namespace Jetstream {

class Instance;

class JETSTREAM_API Flowgraph {
 public:
    explicit Flowgraph(Instance& instance);
    ~Flowgraph();

    Result create();
    Result create(const std::string& path);
    Result create(const char* blob);
    Result destroy();

    Result exportToFile();
    Result exportToBlob();

    Result print() const;

    U64 empty() const {
        return _nodes.empty();
    }

    constexpr const bool& imported() const {
        return _imported;
    }

    constexpr const std::vector<char>& blob() const {
        return _blob;
    }

    constexpr const std::string& filename() const {
        return _filename;
    }
    Result setFilename(const std::string& filename);

    constexpr const std::string& protocolVersion() const {
        return _protocolVersion;
    }

    constexpr const std::string& cyberetherVersion() const {
        return _cyberetherVersion;
    }

    constexpr const std::string& title() const {
        return _title;
    }
    Result setTitle(const std::string& title);

    constexpr const std::string& summary() const {
        return _summary;
    }
    Result setSummary(const std::string& summary);

    constexpr const std::string& author() const {
        return _author;
    }
    Result setAuthor(const std::string& author);

    constexpr const std::string& license() const {
        return _license;
    }
    Result setLicense(const std::string& license);

    constexpr const std::string& description() const {
        return _description;
    }
    Result setDescription(const std::string& description);

    // Manifest, and metadata.

    struct Metadata {
        std::string title;
        std::string summary;
        std::string description;
        const char* data;
    };

    typedef std::map<std::string, Metadata> MetadataManifest;

 protected:
    // Nodes.

    struct Node {
        std::string id;

        Parser::RecordMap inputMap;
        Parser::RecordMap outputMap;
        Parser::RecordMap configMap;
        Parser::RecordMap stateMap;

        Block::Fingerprint fingerprint;

        std::shared_ptr<Block> block;
        std::shared_ptr<Module> module;

        std::shared_ptr<Compute> compute;
        std::shared_ptr<Present> present;

        void setConfigEndpoint(auto& endpoint) {
            getConfigFunc = [&](Parser::RecordMap& map){ return endpoint >> map; };
        }

        void setStateEndpoint(auto& endpoint) {
            getStateFunc = [&](Parser::RecordMap& map){ return endpoint >> map; };
        }

        Result updateMaps() {
            if (getConfigFunc) {
                JST_CHECK(getConfigFunc(configMap));
            }

            if (getStateFunc) {
                JST_CHECK(getStateFunc(stateMap));
            }

            return Result::SUCCESS;
        }

     private:
        std::function<Result(Parser::RecordMap&)> getConfigFunc;
        std::function<Result(Parser::RecordMap&)> getStateFunc;
    };

    typedef std::unordered_map<Locale, std::shared_ptr<Node>, Locale::Hasher> Nodes;

    constexpr const Nodes& nodes() const {
        return _nodes;
    }

    constexpr Nodes& nodes() {
        return _nodes;
    }

    constexpr const std::vector<Locale>& nodesOrder() const {
        return _nodesOrder;
    }

    constexpr std::vector<Locale>& nodesOrder() {
        return _nodesOrder;
    }

    friend class Instance;

 private:
    Instance& _instance;

    Nodes _nodes;
    std::vector<Locale> _nodesOrder;

    struct YamlImpl;
    std::unique_ptr<YamlImpl> _yaml;

    std::string _filename;
    std::vector<char> _blob;
    bool _imported = false;

    std::string _protocolVersion;
    std::string _cyberetherVersion;
    std::string _title;
    std::string _summary;
    std::string _author;
    std::string _license;
    std::string _description;

    Result importFromBlob();    
};

}  // namespace Jetstream

#endif
