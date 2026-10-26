#ifndef JETSTREAM_DOMAINS_IO_UDP_TRANSMITTER_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_UDP_TRANSMITTER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct UdpTransmitter : public Block::Config {
    std::string address = "127.0.0.1";
    U64 port = 5000;
    std::string networkInterface = "auto";
    U64 packetSize = 1472;

    JST_BLOCK_TYPE(udp_transmitter);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_PARAMS(address, port, networkInterface, packetSize);
    JST_BLOCK_DESCRIPTION(
        "UDP Transmitter",
        "Sends data streams over UDP.",
        "# UDP Transmitter\n"
        "\n"
        "Sends each incoming buffer as raw bytes in datagrams to an address and port. "
        "It usually ends a chain, streaming samples or results to another program or to"
        " a UDP Receiver on another machine.\n"
        "\n"
        "- **Each buffer is split into packets.** Every packet carries whole samples, "
        "up to Packet Size bytes.\n"
        "- **Delivery is not guaranteed.** Lost packets are not resent, and the block "
        "never waits for a reader.\n"
        "- **The stream has no header.** Type, shape, and sample rate are not sent, so "
        "the reader must know them.\n"
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
        "| **Address** | 127.0.0.1 | An address or host name, broadcast and multicast "
        "included | Destination of the packets. |\n"
        "| **Port** | 5000 | 1 to 65535 | Destination port. |\n"
        "| **Interface** | Automatic | Automatic and the network interfaces found | "
        "Interface that multicast packets leave from. Automatic lets the system choose."
        " Unicast and broadcast packets always follow the routing table. |\n"
        "| **Packet Size** | 1,472 bytes | 1 to 65,507 bytes, and at least one sample |"
        " Largest payload per packet, rounded down to whole samples. The default fits a"
        " standard Ethernet frame. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **Throughput** | Smoothed send rate in MB/s. |\n"
        "| **Dropped Packets** | Packets that failed to send since the block started. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume a reader listening at the destination.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Stream radio samples | Defaults | 8,192 complex samples at 2 MS/s | 45 "
        "packets per buffer, at 16 MB/s |\n"
        "| Feed a UDP Receiver | Defaults, with the receiver on CF32 | 8 batches x "
        "8,192 complex samples | 357 packets per buffer, read back as the same batches "
        "|\n"
        "| Jumbo frame network | Packet Size 8,972 bytes | 8,192 complex samples | 8 "
        "packets of up to 8,968 bytes per buffer |\n"
        "| Broadcast demodulated audio | Address 192.168.1.255 | 819 real samples at "
        "200 kS/s | 3 packets per buffer to every host on the subnet |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that it failed to open a socket | Address cannot be "
        "resolved, or Interface is missing or down on this machine | Check the spelling"
        " of Address, or pick an Interface again. |\n"
        "| Dropped Packets keeps rising | The destination is unreachable, or the chain "
        "sends faster than the network carries | Check the route to Address, or lower "
        "the rate. |\n"
        "| The reader receives nothing | A firewall blocks the port, or the reader "
        "listens on another port | Allow the port through the firewall, and match Port "
        "on both ends. |\n"
        "| The reader sees noise | The reader expects another type, or it lost packets "
        "| Match the type on both ends, and lower the rate on lossy links. |\n"
        "| Large packets never arrive | Packets larger than a network frame are split "
        "in transit, and some networks drop the pieces | Keep Packet Size at the "
        "default. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each buffer is cut into packets and sent during the compute step, so a slow "
        "network stack slows the whole chain. A packet that cannot be sent within a "
        "short timeout is counted as dropped and skipped.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- Packets go back into a chain through a **UDP Receiver**.\n"
        "- **File Writer** records the same raw bytes to a file.\n"
        "- Radio samples to send come from **SDR**."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_UDP_TRANSMITTER_BLOCK_HH
