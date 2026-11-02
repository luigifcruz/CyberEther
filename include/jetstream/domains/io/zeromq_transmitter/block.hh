#ifndef JETSTREAM_DOMAINS_IO_ZEROMQ_TRANSMITTER_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_ZEROMQ_TRANSMITTER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct ZeromqTransmitter : public Block::Config {
    std::string pattern = "publish";
    std::string mode = "bind";
    std::string address = "0.0.0.0";
    U64 port = 5555;
    bool sendAttributes = true;

    JST_BLOCK_TYPE(zeromq_transmitter);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_PARAMS(pattern, mode, address, port, sendAttributes);
    JST_BLOCK_DESCRIPTION(
        "ZeroMQ Transmitter",
        "Sends data streams over ZeroMQ.",
        "# ZeroMQ Transmitter\n"
        "\n"
        "Sends each incoming buffer as one ZeroMQ message, publishing it to every "
        "subscriber or pushing it to one puller at a time. It usually ends a chain, "
        "streaming samples with their sample rate and frequency to a ZeroMQ Receiver or"
        " another program.\n"
        "\n"
        "- **Attributes travel with the samples.** A header part carries type, shape, "
        "and attributes such as `sampleRate` and `frequency`.\n"
        "- **Turn off Send Attributes for other programs.** Programs that read every "
        "message part as samples need plain single-part messages.\n"
        "- **Sending never waits.** Publish drops a buffer for a slow subscriber, and "
        "Push drops it when no puller is ready.\n"
        "- **Either side can listen.** Bind waits for receivers, and Connect reaches "
        "out and retries twice a second.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | Any type | Any shape, sent in memory order. Complex values "
        "are sent as real then imaginary parts. The buffer must be contiguous. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Pattern** | Publish | Publish, Push | Publish sends every buffer to every "
        "subscriber. Push hands each buffer to one puller in turn. |\n"
        "| **Mode** | Bind | Bind, Connect | Listen for receivers, or connect to one "
        "receiver and reconnect by itself. |\n"
        "| **Address** | 0.0.0.0 | An address, or a host name in Connect mode | Local "
        "address to listen on, where 127.0.0.1 keeps it private to this machine. In "
        "Connect mode, the receiver to connect to. |\n"
        "| **Port** | 5555 | 1 to 65535 | Local port to listen on. In Connect mode, the"
        " port of the receiver. |\n"
        "| **Send Attributes** | On | On, Off | Sends a header part with type, shape, "
        "and attributes ahead of each buffer. Turn it off for programs that expect "
        "plain samples. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **Throughput** | Smoothed rate in MB/s that ZeroMQ accepted for sending. "
        "Publish counts buffers even with no subscriber. |\n"
        "| **Peers** | Receivers connected right now. |\n"
        "| **Dropped Buffers** | Buffers dropped because no puller was ready. Push "
        "only. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Feed a ZeroMQ Receiver | Defaults, receiver on its defaults | 8 batches x "
        "8,192 complex samples | One message per buffer with a header and 512 KiB of "
        "samples |\n"
        "| Feed a plain ZeroMQ program | Send Attributes Off | 8,192 complex samples at"
        " 2 MS/s | 244 single-part messages per second of 64 KiB each |\n"
        "| Share work across pullers | Push | Any buffer | Each buffer goes to the next"
        " ready puller |\n"
        "| Push to a listening receiver | Connect, Address 10.0.0.5 | Any buffer | "
        "Every buffer to the receiver at that host on port 5555 |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that it failed to listen | Another program uses Port, or "
        "Address is not on this machine | Close the other program, or pick another Port"
        " or Address. |\n"
        "| Peers stays at 0 | No receiver connects, a firewall blocks the port, or the "
        "receiver uses another port | Start the receiver, allow the port through the "
        "firewall, and match Port on both ends. |\n"
        "| The log warns that a peer disconnected during the handshake | The two ends "
        "use patterns that do not pair, such as Publish with Pull | Pair Publish with "
        "Subscribe, or Push with Pull. |\n"
        "| The log warns that no receiver is ready | Push has no connected puller, or "
        "every puller is full | Start a receiver, or lighten its chain. |\n"
        "| Another program sees noise | It reads the header part as samples | Turn off "
        "Send Attributes. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each buffer becomes a ZeroMQ message during the compute step, and ZeroMQ's own"
        " threads deliver it, so a slow receiver never stalls the chain. The header is "
        "a small JSON object with a version, a stream identifier, and a sequence number"
        " that lets receivers count lost buffers. The connection is plain TCP without "
        "encryption, and the block is not available in the browser.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- Messages go back into a chain through a **ZeroMQ Receiver**.\n"
        "- Without a connection, a **UDP Transmitter** sends the same raw bytes.\n"
        "- For one plain byte stream, use a **TCP Transmitter**."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_ZEROMQ_TRANSMITTER_BLOCK_HH
