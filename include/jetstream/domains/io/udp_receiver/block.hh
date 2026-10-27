#ifndef JETSTREAM_DOMAINS_IO_UDP_RECEIVER_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_UDP_RECEIVER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct UdpReceiver : public Block::Config {
    std::string address = "0.0.0.0";
    U64 port = 5000;
    std::string networkInterface = "auto";
    bool filterSource = false;
    std::string sourceAddress = "";
    std::string dataType = "CF32";
    U64 numberOfBatches = 8;
    U64 numberOfTimeSamples = 8192;
    U64 bufferMultiplier = 4;

    JST_BLOCK_TYPE(udp_receiver);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_PARAMS(address, port, networkInterface, filterSource,
                     sourceAddress, dataType, numberOfBatches,
                     numberOfTimeSamples, bufferMultiplier);
    JST_BLOCK_DESCRIPTION(
        "UDP Receiver",
        "Receives data streams over UDP.",
        "# UDP Receiver\n"
        "\n"
        "Receives raw samples from datagrams sent to a local port and cuts them into "
        "buffers of batches. It usually starts a chain in place of a radio, fed by a "
        "program that streams samples over the network.\n"
        "\n"
        "- **The sender paces the chain.** Each buffer comes out only once enough bytes"
        " have arrived.\n"
        "- **Datagrams join into one byte stream.** Packet boundaries are ignored, so "
        "each datagram should carry whole samples.\n"
        "- **Lost packets are not detected.** A dropped datagram leaves a silent gap in"
        " the stream.\n"
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
        "| **Address** | 0.0.0.0 | A local address, a host name, or a multicast group |"
        " Local address to listen on. The default covers every IPv4 interface. A "
        "multicast group address joins that group. |\n"
        "| **Port** | 5000 | 1 to 65535 | Local port to listen on. |\n"
        "| **Interface** | Automatic | Automatic and the network interfaces found | "
        "Interface to join multicast groups on, chosen by the system for Automatic. "
        "Unicast and broadcast traffic is received on every interface. |\n"
        "| **Filter Source** | Off | On, Off | Drops packets from every host except "
        "Source Address. |\n"
        "| **Source Address** | Empty | An IPv4 or IPv6 address, or a host name | Only "
        "sender to accept, matched by address on any port. Shown while Filter Source is"
        " on. |\n"
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
        "| **Throughput** | Rate the chain reads in MB/s. |\n"
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
        "| Multicast feed | Address 239.1.1.1 | 8 batches x 8,192 complex samples every"
        " 32.8 ms, shared with every receiver in the group |\n"
        "| One sender on a shared port | Filter Source on, Source Address 10.0.0.5 | 8 "
        "batches x 8,192 complex samples every 32.8 ms, from that host only |\n"
        "| Low latency | Batches 1, Samples 1,024 | 1,024 complex samples every 0.51 "
        "ms, at 16 MB/s |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that it failed to listen | Another program uses Port, "
        "Address is not on this machine, or Interface is missing or down | Close the "
        "other program, or pick another Port, Address, or Interface. |\n"
        "| The output never updates | Nothing reaches Port, such as when a firewall "
        "blocks it, or Filter Source drops the sender | Check the sender destination "
        "and Source Address, and allow the port through the firewall. |\n"
        "| The output looks like noise | Data Type does not match the sender format, or"
        " datagrams split samples | Set Data Type to the sender format, and send whole "
        "samples per datagram. |\n"
        "| Glitches appear at high rates | The operating system drops datagrams when "
        "its receive buffer fills | Lower the rate, or raise the system receive buffer "
        "limit. |\n"
        "| Buffer Loss rises above 0% | The chain reads slower than the sender sends, "
        "or one burst outgrows the internal buffer | Lighten the chain. Raise Buffer "
        "Multiplier for short stalls and large bursts. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "A background thread appends each datagram to a ring buffer, and each cycle "
        "copies out one full buffer once enough bytes have arrived. When the ring "
        "buffer is full, new bytes overwrite the oldest, which Buffer Loss counts. The "
        "socket asks the operating system for a large receive buffer to ride out short "
        "stalls.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- Samples from a **UDP Transmitter** arrive here as sent.\n"
        "- **WebSocket Receiver** receives the same raw samples over a connection.\n"
        "- **Cast** scales integer samples to F32 or CF32."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_UDP_RECEIVER_BLOCK_HH
