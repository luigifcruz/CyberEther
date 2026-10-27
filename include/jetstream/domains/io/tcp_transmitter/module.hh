#ifndef JETSTREAM_DOMAINS_IO_TCP_TRANSMITTER_MODULE_HH
#define JETSTREAM_DOMAINS_IO_TCP_TRANSMITTER_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct TcpTransmitter : public Module::Config {
    std::string mode = "server";
    std::string address = "0.0.0.0";
    U64 port = 5001;

    JST_MODULE_TYPE(tcp_transmitter);
    JST_MODULE_PARAMS(mode, address, port);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_TCP_TRANSMITTER_MODULE_HH
