#ifndef JETSTREAM_DETAIL_FLOWGRAPH_IMPL_HH
#define JETSTREAM_DETAIL_FLOWGRAPH_IMPL_HH

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "jetstream/flowgraph_environment.hh"
#include "jetstream/flowgraph_metadata.hh"

namespace Jetstream {

struct Flowgraph::Impl {
    struct EnvironmentEntry {
        U64 start;
        U64 end;
        U64 sequence;
        Parser::Map data;
    };

    bool created = false;

    std::unique_ptr<Metadata> metadata;
    std::shared_ptr<Environment> environment;

    std::shared_ptr<Instance> instance;
    std::shared_ptr<Render::Window> render;
    std::shared_ptr<Compositor> compositor;

    std::shared_ptr<Scheduler> scheduler;
    std::unordered_map<std::string, std::shared_ptr<Block>> blocks;
    std::vector<std::string> blockOrder;
    std::unordered_map<std::string, std::vector<std::string>> edges;

    std::string title;
    std::string summary;
    std::string author;
    std::string license;
    std::string description;
    std::string path;

    Parser::Map metadataValues;
    std::unordered_map<std::string, Parser::Map> blockMetadataValues;

    mutable std::shared_mutex environmentMutex;
    std::unordered_map<std::string, std::vector<EnvironmentEntry>> environmentValues;
    U64 environmentSequence = 0;

    Result resolveInputs(const TensorMap& requested, TensorMap& resolved) const;
    std::vector<std::string> collectDownstream(const std::string& name) const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_DETAIL_FLOWGRAPH_IMPL_HH
