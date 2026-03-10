#ifndef JETSTREAM_DOMAINS_IO_WEBSOCKET_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_WEBSOCKET_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Websocket : public Block::Config {
    std::string url = "ws://localhost:8765";
    std::string dataType = "CF32";
    U64 numberOfBatches = 8;
    U64 numberOfTimeSamples = 8192;
    U64 bufferMultiplier = 4;

    JST_BLOCK_TYPE(websocket);
    JST_BLOCK_PARAMS(url, dataType, numberOfBatches, numberOfTimeSamples,
                     bufferMultiplier);
    JST_BLOCK_DESCRIPTION(
        "WebSocket",
        "Receives data streams over WebSocket.",
        "# WebSocket Client\n"
        "The WebSocket block receives data streams from a WebSocket server, "
        "enabling browser-based signal processing pipelines fed by external "
        "data sources.\n\n"

        "## Arguments\n"
        "- **URL**: WebSocket server URL.\n"
        "- **Data Type**: Sample format of incoming data.\n"
        "- **Number of Batches**: Number of batches in output buffer.\n"
        "- **Number of Time Samples**: Samples per batch.\n"
        "- **Buffer Multiplier**: Internal buffer size multiplier.\n\n"

        "## Useful For\n"
        "- Receiving real or complex RF samples in the browser from a remote source.\n"
        "- Browser-based spectrum analysis with external data feeds.\n"
        "- Connecting browser pipelines to server-side SDR devices.\n\n"

        "## Examples\n"
        "- Receive IQ samples from local server:\n"
        "  Config: URL=ws://localhost:8765, Data Type=CF32, Batches=8, Samples=8192\n"
        "  Output: CF32[8, 8192]\n"
        "- Receive U16 sample batches from local server:\n"
        "  Config: URL=ws://localhost:8765, Data Type=U16, Batches=4, Samples=4096\n"
        "  Output: U16[4, 4096]\n\n"

        "## Implementation\n"
        "WebSocket Module -> Output Buffer\n"
        "1. Opens WebSocket connection to specified URL.\n"
        "2. Receives binary frames containing samples.\n"
        "3. Writes samples into circular buffer for downstream processing."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_WEBSOCKET_BLOCK_HH
