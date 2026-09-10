#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_WATERFALL_HISTORY_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_WATERFALL_HISTORY_HH

#include <algorithm>

#include <jetstream/types.hh>

namespace Jetstream::Modules {

struct WaterfallAveragingPlan {
    U64 rowCount = 0;
    U64 pendingRows = 0;
};

inline WaterfallAveragingPlan PlanWaterfallAveraging(const U64 pendingRows,
                                                    const U64 incomingRows,
                                                    const U64 averaging) {
    const U64 neededRows = averaging - pendingRows;
    if (incomingRows < neededRows) {
        return {.rowCount = 0, .pendingRows = pendingRows + incomingRows};
    }
    const U64 remainingRows = incomingRows - neededRows;
    return {
        .rowCount = 1 + remainingRows / averaging,
        .pendingRows = remainingRows % averaging,
    };
}

struct WaterfallWritePlan {
    U64 sourceRow = 0;
    U64 destinationRow = 0;
    U64 rowCount = 0;
};

inline WaterfallWritePlan PlanWaterfallWrite(const U64 writeIndex,
                                             const U64 numberOfBatches,
                                             const U64 height) {
    const U64 retainedRows = std::min(numberOfBatches, height);
    const U64 sourceRow = numberOfBatches - retainedRows;
    return {
        .sourceRow = sourceRow,
        .destinationRow = (writeIndex + (sourceRow % height)) % height,
        .rowCount = retainedRows,
    };
}

struct WaterfallDirtyPlan {
    U64 startRow = 0;
    U64 firstRowCount = 0;
    U64 secondRowCount = 0;
};

struct WaterfallHistory {
    U64 writeIndex = 0;
    U64 dirtyRows = 0;

    void advance(const U64 numberOfBatches, const U64 height) {
        writeIndex = (writeIndex + (numberOfBatches % height)) % height;
        dirtyRows += std::min(numberOfBatches, height - dirtyRows);
    }

    WaterfallDirtyPlan dirtyPlan(const U64 height) const {
        const U64 startRow = (writeIndex + height - dirtyRows) % height;
        const U64 firstRowCount = std::min(dirtyRows, height - startRow);
        return {
            .startRow = startRow,
            .firstRowCount = firstRowCount,
            .secondRowCount = dirtyRows - firstRowCount,
        };
    }

    void clearDirty() {
        dirtyRows = 0;
    }
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_WATERFALL_HISTORY_HH
