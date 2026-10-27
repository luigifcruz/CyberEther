#ifndef JETSTREAM_DOMAINS_IO_WEBSOCKET_TRANSMITTER_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_WEBSOCKET_TRANSMITTER_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct WebsocketTransmitter : public Block::Config {
    std::string mode = "server";
    std::string address = "0.0.0.0";
    U64 port = 8765;
    std::string url = "ws://localhost:8765";

    JST_BLOCK_TYPE(websocket_transmitter);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_PARAMS(mode, address, port, url);
    JST_BLOCK_DESCRIPTION(
        "WebSocket Transmitter",
        "Sends data streams over WebSocket.",
        "# WebSocket Transmitter\n"
        "\n"
        "Sends each incoming buffer as one binary WebSocket message, either hosting a "
        "server or connecting to one. It usually ends a chain, streaming samples to a "
        "WebSocket Receiver on another machine or to a browser.\n"
        "\n"
        "- **One buffer is one message.** The whole buffer goes out as raw bytes in a "
        "single binary message.\n"
        "- **Every client gets every buffer.** Clients connect on any path, and a late "
        "client starts at the next buffer.\n"
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
        "are sent as real then imaginary parts. The buffer must be contiguous and at "
        "most 16 MiB. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Mode** | Server | Server or Client | Host a server, or connect to one. |\n"
        "| **Address** | 0.0.0.0 | A local address or a host name | Local address to "
        "listen on. The default covers every IPv4 interface, and 127.0.0.1 keeps the "
        "server private to this machine. Server mode only. |\n"
        "| **Port** | 8765 | 1 to 65535 | Local port to listen on. Server mode only. |\n"
        "| **URL** | ws://localhost:8765 | A ws:// or wss:// address with a host and no"
        " # fragment | Server to send to. Client mode only. |\n"
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
        "| Feed a WebSocket Receiver | Defaults, receiver on ws://host:8765 | 8 batches"
        " x 8,192 complex samples | One message per buffer, read back as the same "
        "batches |\n"
        "| Stream radio samples | Defaults | 8,192 complex samples at 2 MS/s | 244 "
        "messages per second of 64 KiB each, at 16 MB/s per client |\n"
        "| Local only | Address 127.0.0.1 | Any buffer | Reachable only from this "
        "machine |\n"
        "| Send through a relay | Mode Client, URL ws://relay:8765/stream | Any buffer "
        "| One message per buffer to the relay, which forwards it to every receiver on "
        "that path |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that it failed to listen | Another program uses Port, or "
        "Address is not on this machine | Close the other program, or pick another Port"
        " or Address. |\n"
        "| Clients stays at 0 | A firewall blocks the port, or the client uses another "
        "port | Allow the port through the firewall, and match Port in the client URL. "
        "|\n"
        "| Connection stays at Connecting | The server is down, or URL is wrong | Start"
        " the server, and check the host, port, and scheme in URL. |\n"
        "| The chain slows down while a client is connected | That client reads slower "
        "than the chain sends | Lower the rate, or disconnect the slow client. |\n"
        "| The block reports that the input exceeds the message limit | One buffer is "
        "larger than 16 MiB | Lower Batches or Samples upstream. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each buffer is sent during the compute step, so a slow peer slows the whole "
        "chain, and the server disconnects a client that stops reading. The server is "
        "plain ws without encryption and is not available in the browser, where Client "
        "mode sends through a relay instead. Client mode accepts wss without checking "
        "the certificate, and drops any message the server sends back.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- Messages go back into a chain through a **WebSocket Receiver**.\n"
        "- Without a connection, a **UDP Transmitter** sends the same raw bytes.\n"
        "- **File Writer** records the same raw bytes to a file."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_WEBSOCKET_TRANSMITTER_BLOCK_HH
