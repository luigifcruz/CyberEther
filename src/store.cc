#include "jetstream/store.hh"

#include "jetstream/blocks/base.hh"
#include "jetstream/modules/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/viewport/base.hh"
#include "jetstream/backend/base.hh"

#include "examples/flowgraphs/manifest.hh"
#include "jetstream/blocks/manifest.hh"

namespace Jetstream {

Store& Store::GetInstance() {
    static Store store;
    return store;
}

Store::Store() {
    Blocks::GetDefaultManifest(blockConstructorList, blockMetadataList);
    Flowgraphs::GetDefaultManifest(flowgraphMetadataList);
}

Result Store::_blockList(const std::string& filter) {
    // Return if the filter is the same as last time.
    if (filter == lastBlockFilter &&
        !blockFilteredMetadataList.empty()) {
        return Result::SUCCESS;
    }
    lastBlockFilter = filter;

    // Apply filter logic.
    blockFilteredMetadataList.clear();

    std::string filterLower = filter;
    std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(), ::tolower);

    for (const auto& [key, value] : blockMetadataList) {
        std::string titleLower = value.title;
        std::transform(titleLower.begin(), titleLower.end(), titleLower.begin(), ::tolower);

        std::string smallLower = value.summary;
        std::transform(smallLower.begin(), smallLower.end(), smallLower.begin(), ::tolower);

        std::string detailedLower = value.description;
        std::transform(detailedLower.begin(), detailedLower.end(), detailedLower.begin(), ::tolower);

        // Case-insensitive search.
        if (titleLower.find(filterLower) != std::string::npos ||
            smallLower.find(filterLower) != std::string::npos ||
            detailedLower.find(filterLower) != std::string::npos ||
            filter.empty()) {
            blockFilteredMetadataList[key] = value;
        }
    }

    return Result::SUCCESS;
}

Result Store::_flowgraphList(const std::string& filter) {
    // Return if the filter is the same as last time.
    if (filter == lastFlowgraphFilter &&
        !filteredFlowgraphMetadataList.empty()) {
        return Result::SUCCESS;
    }
    lastFlowgraphFilter = filter;

    // Apply filter logic.
    filteredFlowgraphMetadataList.clear();

    std::string filterLower = filter;
    std::transform(filterLower.begin(), filterLower.end(), filterLower.begin(), ::tolower);

    for (const auto& [key, value] : flowgraphMetadataList) {
        std::string titleLower = value.title;
        std::transform(titleLower.begin(), titleLower.end(), titleLower.begin(), ::tolower);

        std::string descriptionLower = value.description;
        std::transform(descriptionLower.begin(), descriptionLower.end(), descriptionLower.begin(), ::tolower);

        // Case-insensitive search.
        if (titleLower.find(filterLower) != std::string::npos ||
            descriptionLower.find(filterLower) != std::string::npos) {
            filteredFlowgraphMetadataList[key] = value;
        }
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream
