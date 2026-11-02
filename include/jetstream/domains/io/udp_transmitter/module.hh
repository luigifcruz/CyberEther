#ifndef JETSTREAM_DOMAINS_IO_UDP_TRANSMITTER_MODULE_HH
#define JETSTREAM_DOMAINS_IO_UDP_TRANSMITTER_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct UdpTransmitter : public Module::Config {
    std::string address = "127.0.0.1";
    U64 port = 5000;
    std::string networkInterface = "auto";
    U64 packetSize = 1472;

    JST_MODULE_TYPE(udp_transmitter);
    JST_MODULE_PARAMS(address, port, networkInterface, packetSize);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_UDP_TRANSMITTER_MODULE_HH
