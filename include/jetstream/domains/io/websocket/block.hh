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
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_PARAMS(url, dataType, numberOfBatches, numberOfTimeSamples,
                     bufferMultiplier);
    JST_BLOCK_DESCRIPTION(
        "WebSocket",
        "Receives data streams over WebSocket.",
        "# WebSocket\n"
        "\n"
        "Receives raw samples as binary messages from a WebSocket server and cuts them "
        "into buffers of batches. It usually starts a chain in place of a radio, fed by"
        " a server that streams samples from a remote receiver.\n"
        "\n"
        "- **The server paces the chain.** Each buffer comes out only once the server "
        "has sent all of its bytes.\n"
        "- **Messages join into one byte stream.** Message boundaries are ignored, and "
        "text messages are dropped.\n"
        "- **Samples come out as sent.** Integer types are not scaled, so convert them "
        "with a Cast block.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Output** (out) | Data Type, CF32 by default | Batches x Samples, on a "
        "`batchAxis` then a `sampleAxis`. Neither `sampleRate` nor `frequency` is set. "
        "|\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **URL** | ws://localhost:8765 | A ws:// or wss:// address with a host and no"
        " # fragment | Server to connect to. |\n"
        "| **Data Type** | CF32 | CF32, F32, CI8, I8, CU8, U8, CI16, I16, CU16, U16 | "
        "Type of each received sample, and of the output. It must match what the server"
        " sends. |\n"
        "| **Batches** | 8 batches | 1 or more | Batches per output buffer. |\n"
        "| **Samples** | 8,192 samples | 1 or more | Samples per batch, and the FFT "
        "size downstream. |\n"
        "| **Buffer Multiplier** | 4x | 1 or more | Output buffers the internal buffer "
        "holds before old samples are overwritten. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **Buffer Health** | How full the internal buffer is, smoothed, from 0 to "
        "100%. Near 100% means the chain is falling behind. |\n"
        "| **Throughput** | Rate the chain reads in MB/s, refreshed as messages arrive."
        " |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume a server streaming 2 MS/s in the chosen Data Type.\n"
        "\n"
        "| Goal | Settings | Output |\n"
        "|---|---|---|\n"
        "| Complex stream | Defaults | 8 batches x 8,192 complex samples every 32.8 ms,"
        " at 16 MB/s |\n"
        "| Eight-bit radio samples | CI8 | 8 batches x 8,192 complex samples every 32.8"
        " ms, at 4 MB/s, not scaled |\n"
        "| Real 16-bit batches | U16, Batches 4, Samples 4,096 | 4 batches x 4,096 real"
        " samples every 8.2 ms, at 4 MB/s |\n"
        "| Low latency | Batches 1, Samples 1,024 | 1,024 complex samples every 0.51 "
        "ms, at 16 MB/s |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that it failed to connect | No server answers at URL when "
        "the block starts | Start the server, then edit URL or reopen the flowgraph. |\n"
        "| The output stops updating and the log warns that the connection closed | The"
        " server dropped the connection, and the block does not reconnect | Restart the"
        " server, then reopen the flowgraph. |\n"
        "| The block reports an invalid WebSocket URL | The URL lacks ws:// or wss://, "
        "lacks a host, or has a # fragment | Write it as ws://host:port/path. |\n"
        "| The output looks like noise | Data Type does not match the format the server"
        " sends | Set Data Type to the server format. |\n"
        "| Buffer Health stays near 100% | The chain reads slower than the server "
        "sends, so the oldest samples are overwritten | Lighten the chain. Raise Buffer"
        " Multiplier for short stalls. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Incoming binary messages are appended to a ring buffer, and each cycle copies "
        "out one full buffer once enough bytes have arrived. When the ring buffer is "
        "full, new bytes overwrite the oldest without a warning. Outside the browser, "
        "secure connections skip certificate checks.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **File Reader** reads the same raw samples from a file.\n"
        "- **Cast** scales integer samples to F32 or CF32.\n"
        "- **Spectrum Analyzer** shows what the server sends."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_WEBSOCKET_BLOCK_HH
