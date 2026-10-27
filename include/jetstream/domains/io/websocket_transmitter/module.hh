#ifndef JETSTREAM_DOMAINS_IO_WEBSOCKET_TRANSMITTER_MODULE_HH
#define JETSTREAM_DOMAINS_IO_WEBSOCKET_TRANSMITTER_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct WebsocketTransmitter : public Module::Config {
    std::string mode = "server";
    std::string address = "0.0.0.0";
    U64 port = 8765;
    std::string url = "ws://127.0.0.1:8765";

    JST_MODULE_TYPE(websocket_transmitter);
    JST_MODULE_PARAMS(mode, address, port, url);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_WEBSOCKET_TRANSMITTER_MODULE_HH
