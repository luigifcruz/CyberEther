#ifndef JETSTREAM_DOMAINS_IO_TCP_TRANSMITTER_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_TCP_TRANSMITTER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct TcpTransmitter : public Block::Config {
    std::string mode = "server";
    std::string address = "0.0.0.0";
    U64 port = 5001;

    JST_BLOCK_TYPE(tcp_transmitter);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_PARAMS(mode, address, port);
    JST_BLOCK_DESCRIPTION(
        "TCP Transmitter",
        "Sends data streams over TCP.",
        "# TCP Transmitter\n"
        "\n"
        "Sends each incoming buffer as raw bytes over TCP, either serving every client "
        "that connects or connecting to one receiver. It usually ends a chain, "
        "streaming samples to a TCP Receiver or another program on the network.\n"
        "\n"
        "- **A stalled connection is closed.** A buffer cut short by a stalled send is "
        "the last one that connection gets.\n"
        "- **Every client gets every buffer.** A client that joins late starts at the "
        "next buffer.\n"
        "- **Client mode reconnects.** It retries twice a second, and buffers sent "
        "while disconnected are dropped.\n"
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
        "| **Mode** | Server | Server, Client | Serve every client that connects, or "
        "connect to one receiver. |\n"
        "| **Address** | 0.0.0.0 | An address or a host name | Local address to listen "
        "on, where 127.0.0.1 keeps the server private to this machine. In Client mode, "
        "the receiver to connect to. |\n"
        "| **Port** | 5001 | 1 to 65535 | Local port to listen on. In Client mode, the "
        "port of the receiver. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **Throughput** | Smoothed send rate in MB/s, summed over every client. |\n"
        "| **Clients** | Clients connected right now. Server mode only. |\n"
        "| **Connection** | Connected, or Connecting while it retries. Client mode "
        "only. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Feed a TCP Receiver | Defaults, receiver on its defaults | 8 batches x 8,192"
        " complex samples | 512 KiB per buffer, read back as the same batches |\n"
        "| Stream radio samples | Defaults | 8,192 complex samples at 2 MS/s | 244 "
        "buffers per second of 64 KiB each, at 16 MB/s per client |\n"
        "| Local only | Address 127.0.0.1 | Any buffer | Reachable only from this "
        "machine |\n"
        "| Push to a listening receiver | Client, Address 10.0.0.5 | Any buffer | Every"
        " buffer to the receiver at that host on port 5001 |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that it failed to listen | Another program uses Port, or "
        "Address is not on this machine | Close the other program, or pick another Port"
        " or Address. |\n"
        "| Clients stays at 0 | A firewall blocks the port, or the receiver uses "
        "another port | Allow the port through the firewall, and match Port on both "
        "ends. |\n"
        "| Connection stays at Connecting | No receiver listens at Address and Port | "
        "Start the receiver in Server mode, and check Address and Port. |\n"
        "| The chain slows down while a client is connected | That client reads slower "
        "than the chain sends | Lower the rate, or disconnect the slow client. |\n"
        "| The reader sees noise | The reader expects another type | Match the type on "
        "both ends. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each buffer is sent during the compute step, so a slow peer slows the whole "
        "chain until a stalled write closes its connection. A background thread accepts"
        " clients in Server mode, or keeps the connection open in Client mode and drops"
        " anything the receiver sends back. The connection is plain TCP without "
        "encryption, and the block is not available in the browser.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- Bytes go back into a chain through a **TCP Receiver**.\n"
        "- Without a connection, a **UDP Transmitter** sends the same raw bytes.\n"
        "- **WebSocket Transmitter** also reaches browsers through a relay."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_TCP_TRANSMITTER_BLOCK_HH
