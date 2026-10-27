#ifndef JETSTREAM_DOMAINS_IO_ZEROMQ_RECEIVER_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_ZEROMQ_RECEIVER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct ZeromqReceiver : public Block::Config {
    std::string pattern = "subscribe";
    std::string mode = "connect";
    std::string address = "127.0.0.1";
    U64 port = 5555;
    std::string dataType = "CF32";
    U64 numberOfBatches = 8;
    U64 numberOfTimeSamples = 8192;
    U64 bufferMultiplier = 4;

    JST_BLOCK_TYPE(zeromq_receiver);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_PARAMS(pattern, mode, address, port, dataType, numberOfBatches,
                     numberOfTimeSamples, bufferMultiplier);
    JST_BLOCK_DESCRIPTION(
        "ZeroMQ Receiver",
        "Receives data streams over ZeroMQ.",
        "# ZeroMQ Receiver\n"
        "\n"
        "Receives samples from ZeroMQ messages, by subscribing to a publisher or "
        "pulling from a pusher, and cuts them into buffers of batches. It usually "
        "starts a chain in place of a radio, fed by a ZeroMQ Transmitter or another "
        "program.\n"
        "\n"
        "- **Attributes arrive with the samples.** When the sender adds a header, "
        "`sampleRate`, `frequency`, and other single values come back on the output.\n"
        "- **Plain messages work too.** Messages without a header are read as samples "
        "of Data Type, and leading key parts are skipped.\n"
        "- **Connect mode reconnects.** It retries twice a second until the sender "
        "answers again.\n"
        "- **Samples come out as sent.** Integer types are not scaled, so convert them "
        "with a Cast block.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Output** (out) | Data Type, CF32 by default | Batches x Samples, on a "
        "`batchAxis` then a `sampleAxis`. Carries the single-value attributes of the "
        "sender, such as `sampleRate`, when it adds a header. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Pattern** | Subscribe | Subscribe, Pull | Subscribe takes every buffer a "
        "publisher sends. Pull takes the buffers a pusher hands out. |\n"
        "| **Mode** | Connect | Connect, Bind | Connect to one sender and reconnect by "
        "itself, or listen for any number of senders. |\n"
        "| **Address** | 127.0.0.1 | An address or a host name | Sender to connect to. "
        "In Bind mode, the local address to listen on, where 0.0.0.0 covers every IPv4 "
        "interface. |\n"
        "| **Port** | 5555 | 1 to 65535 | Port of the sender. In Bind mode, the local "
        "port to listen on. |\n"
        "| **Data Type** | CF32 | CF32, F32, CI8, I8, CU8, U8, CI16, I16, CU16, U16 | "
        "Type of each received sample, and of the output. A sender header with another "
        "type is dropped with a warning. |\n"
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
        "| **Buffer Fill** | How full the internal buffer is, smoothed, from 0 to "
        "100%. Near 100% means the chain is falling behind. |\n"
        "| **Buffer Loss** | Share of samples overwritten before the chain read them "
        "since the stream started. |\n"
        "| **Throughput** | Rate the chain reads in MB/s. |\n"
        "| **Peers** | Senders connected right now. |\n"
        "| **Lost Messages** | Buffers that a sender with a header sent but never "
        "arrived, found from gaps in its sequence numbers. Subscribe only. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume a sender streaming 2 MS/s in the chosen Data Type.\n"
        "\n"
        "| Goal | Settings | Output |\n"
        "|---|---|---|\n"
        "| Stream from a ZeroMQ Transmitter | Defaults | 8 batches x 8,192 complex "
        "samples every 32.8 ms, at 16 MB/s, with `sampleRate` and `frequency` |\n"
        "| Read a plain program | Defaults, sender without a header | 8 batches x 8,192"
        " complex samples every 32.8 ms, with no attributes |\n"
        "| Collect from many senders | Bind, Address 0.0.0.0 | 8 batches x 8,192 "
        "complex samples, filled with whole messages from every sender in turn |\n"
        "| Low latency | Batches 1, Samples 1,024 | 1,024 complex samples every 0.51 "
        "ms, at 16 MB/s |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The output never updates and Peers stays at 0 | No sender listens at Address"
        " and Port, or a firewall blocks the port | Start the sender, check Address and"
        " Port, and allow the port through the firewall. |\n"
        "| The log warns that a peer disconnected during the handshake | The sender "
        "uses a pattern that does not pair with Pattern | Pair Subscribe with Publish, "
        "or Pull with Push. |\n"
        "| The log warns that the sender sends another type | The sender header names a"
        " type other than Data Type | Set Data Type to the sender type. |\n"
        "| The output looks like noise | Data Type does not match a sender without a "
        "header | Set Data Type to the sender format. |\n"
        "| Buffer Loss or Lost Messages rises above 0 | The chain reads slower than the"
        " sender sends, or the network drops messages | Lighten the chain. Raise Buffer"
        " Multiplier for short stalls. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "A background thread stores whole messages in a ring buffer and tracks which "
        "attributes came with which bytes. Each cycle copies out one full buffer with "
        "its matching attributes, and a full ring buffer overwrites the oldest bytes, "
        "which Buffer Loss counts. The connection is plain TCP without encryption, and "
        "the block is not available in the browser.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- Samples from a **ZeroMQ Transmitter** arrive here with their attributes.\n"
        "- For one plain byte stream, use a **TCP Receiver**.\n"
        "- **Cast** scales integer samples to F32 or CF32."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_ZEROMQ_RECEIVER_BLOCK_HH
