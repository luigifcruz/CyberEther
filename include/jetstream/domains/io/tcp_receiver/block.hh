#ifndef JETSTREAM_DOMAINS_IO_TCP_RECEIVER_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_TCP_RECEIVER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct TcpReceiver : public Block::Config {
    std::string mode = "client";
    std::string address = "127.0.0.1";
    U64 port = 5001;
    std::string dataType = "CF32";
    U64 numberOfBatches = 8;
    U64 numberOfTimeSamples = 8192;
    U64 bufferMultiplier = 4;

    JST_BLOCK_TYPE(tcp_receiver);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_PARAMS(mode, address, port, dataType, numberOfBatches,
                     numberOfTimeSamples, bufferMultiplier);
    JST_BLOCK_DESCRIPTION(
        "TCP Receiver",
        "Receives data streams over TCP.",
        "# TCP Receiver\n"
        "\n"
        "Receives raw samples from a TCP connection and cuts them into buffers of "
        "batches. It usually starts a chain in place of a radio, fed by a TCP "
        "Transmitter or another program on the network.\n"
        "\n"
        "- **The sender paces the chain.** Each buffer comes out only once enough bytes"
        " have arrived.\n"
        "- **Client mode reconnects.** It retries twice a second until the sender "
        "answers again.\n"
        "- **Server mode takes one sender at a time.** Other senders wait until the "
        "current one disconnects.\n"
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
        "| **Mode** | Client | Client, Server | Connect to a sender, or listen for one."
        " |\n"
        "| **Address** | 127.0.0.1 | An address or a host name | Sender to connect to. "
        "In Server mode, the local address to listen on, where 0.0.0.0 covers every "
        "IPv4 interface. |\n"
        "| **Port** | 5001 | 1 to 65535 | Port of the sender. In Server mode, the local"
        " port to listen on. |\n"
        "| **Data Type** | CF32 | CF32, F32, CI8, I8, CU8, U8, CI16, I16, CU16, U16 | "
        "Type of each received sample, and of the output. It must match what the sender"
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
        "| **Buffer Loss** | Share of samples overwritten before the chain read them "
        "since the stream started. |\n"
        "| **Throughput** | Rate the chain reads in MB/s, refreshed as data arrives. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume a sender streaming 2 MS/s in the chosen Data Type.\n"
        "\n"
        "| Goal | Settings | Output |\n"
        "|---|---|---|\n"
        "| Complex stream | Defaults | 8 batches x 8,192 complex samples every 32.8 ms,"
        " at 16 MB/s |\n"
        "| Eight-bit radio samples | CI8 | 8 batches x 8,192 complex samples every 32.8"
        " ms, at 4 MB/s, not scaled |\n"
        "| Wait for a sender | Server, Address 0.0.0.0 | 8 batches x 8,192 complex "
        "samples every 32.8 ms from the first sender to connect |\n"
        "| Low latency | Batches 1, Samples 1,024 | 1,024 complex samples every 0.51 "
        "ms, at 16 MB/s |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The output never updates and the log warns that it failed to connect | No "
        "sender listens at Address and Port | Start the sender, or switch Mode to "
        "Server when the sender connects out. |\n"
        "| The block reports that it failed to listen | Another program uses Port, or "
        "Address is not on this machine | Close the other program, or pick another Port"
        " or Address. |\n"
        "| The output pauses and the log warns that the connection closed | The sender "
        "closed the connection | Wait for the block to reconnect, or restart the "
        "sender. |\n"
        "| The output looks like noise | Data Type does not match the sender format | "
        "Set Data Type to the sender format. |\n"
        "| Buffer Loss rises above 0% | The chain reads slower than the sender sends, "
        "or one burst outgrows the internal buffer | Lighten the chain. Raise Buffer "
        "Multiplier for short stalls and large bursts. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "A background thread reads the connection into a ring buffer, which empties "
        "whenever a new connection opens so buffers stay aligned. Each cycle copies out"
        " one full buffer, and when the ring buffer is full, new bytes overwrite the "
        "oldest, which Buffer Loss counts. The connection is plain TCP without "
        "encryption, and the block is not available in the browser.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- Samples from a **TCP Transmitter** arrive here as sent.\n"
        "- Without a connection, a **UDP Receiver** takes the same raw samples.\n"
        "- **Cast** scales integer samples to F32 or CF32."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_TCP_RECEIVER_BLOCK_HH
