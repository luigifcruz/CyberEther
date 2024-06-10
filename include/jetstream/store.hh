#ifndef JETSTREAM_STORE_HH
#define JETSTREAM_STORE_HH

#include "jetstream/parser.hh"
#include "jetstream/instance.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"

namespace Jetstream {

class JETSTREAM_API Store {
 public:
    Store(Store const&) = delete;
    void operator=(Store const&) = delete;

    static Store& GetInstance();

    typedef std::function<Result(Block::ConstructorManifest&, 
                                 Block::MetadataManifest&)> BlockLoaderFunc;

    static Result LoadBlocks(const BlockLoaderFunc& loader) {
        return GetInstance()._loadBlocks(loader);
    }

    static const Block::ConstructorManifest& BlockConstructorList() {
        return GetInstance().blockConstructorList;
    }

    static const Block::MetadataManifest& BlockMetadataList(const std::string& filter = "") {
        GetInstance()._blockList(filter);
        return GetInstance().blockFilteredMetadataList;
    }

    static const Flowgraph::MetadataManifest& FlowgraphMetadataList(const std::string& filter = "") {
        GetInstance()._flowgraphList(filter);
        return GetInstance().filteredFlowgraphMetadataList;
    }

 private:
    Store();

    Result _loadBlocks(const BlockLoaderFunc& loader) {
        JST_CHECK(loader(blockConstructorList, blockMetadataList));
        blockFilteredMetadataList.clear();
        return Result::SUCCESS;
    }

    Block::ConstructorManifest blockConstructorList;
    Block::MetadataManifest blockMetadataList;
    Block::MetadataManifest blockFilteredMetadataList;
    std::string lastBlockFilter;

    Flowgraph::MetadataManifest flowgraphMetadataList;
    Flowgraph::MetadataManifest filteredFlowgraphMetadataList;
    std::string lastFlowgraphFilter;

    Result _blockList(const std::string& filter);
    Result _flowgraphList(const std::string& filter);
};

}  // namespace Jetstream

#endif
