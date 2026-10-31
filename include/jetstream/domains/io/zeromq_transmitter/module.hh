#ifndef JETSTREAM_DOMAINS_IO_ZEROMQ_TRANSMITTER_MODULE_HH
#define JETSTREAM_DOMAINS_IO_ZEROMQ_TRANSMITTER_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct ZeromqTransmitter : public Module::Config {
    std::string pattern = "publish";
    std::string mode = "bind";
    std::string address = "0.0.0.0";
    U64 port = 5555;
    bool sendAttributes = true;

    JST_MODULE_TYPE(zeromq_transmitter);
    JST_MODULE_PARAMS(pattern, mode, address, port, sendAttributes);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_ZEROMQ_TRANSMITTER_MODULE_HH
