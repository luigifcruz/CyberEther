#ifndef JETSTREAM_FLOWGRAPH_DOCUMENT_HH
#define JETSTREAM_FLOWGRAPH_DOCUMENT_HH

#include "jetstream/parser.hh"
#include "jetstream/tensor_link.hh"
#include "jetstream/types.hh"

#include <optional>
#include <string>
#include <vector>

namespace Jetstream {

struct FlowgraphDocument {
    struct Block {
        std::string name;
        std::string module;
        DeviceType device = DeviceType::CPU;
        RuntimeType runtime = RuntimeType::NATIVE;
        ProviderType provider = "generic";
        std::optional<bool> frozen;
        std::optional<Parser::Map> config;
        std::optional<Parser::Map> input;
        std::optional<Parser::Map> meta;

        JST_SERDES(name, module, device, runtime, provider, frozen, config, input, meta);
    };

    std::string version = "2";
    std::optional<std::string> title;
    std::optional<std::string> summary;
    std::optional<std::string> author;
    std::optional<std::string> license;
    std::optional<std::string> description;
    std::vector<Block> graph;
    std::optional<Parser::Map> meta;

    JST_SERDES(version, title, summary, author, license, description, graph, meta);

    Result encode(std::string& yaml) const;

    static std::optional<Parser::Map> CompactMeta(const Parser::Map& meta);
    static std::string Reference(const BlockEndpoint& endpoint);
    static std::optional<BlockEndpoint> ParseReference(const std::string& value);
};

}  // namespace Jetstream

#endif  // JETSTREAM_FLOWGRAPH_DOCUMENT_HH
