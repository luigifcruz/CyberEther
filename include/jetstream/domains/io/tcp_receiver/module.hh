#ifndef JETSTREAM_DOMAINS_IO_TCP_RECEIVER_MODULE_HH
#define JETSTREAM_DOMAINS_IO_TCP_RECEIVER_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct TcpReceiver : public Module::Config {
    std::string mode = "client";
    std::string address = "127.0.0.1";
    U64 port = 5001;
    std::string dataType = "CF32";
    U64 numberOfBatches = 8;
    U64 numberOfTimeSamples = 8192;
    U64 bufferMultiplier = 4;

    JST_MODULE_TYPE(tcp_receiver);
    JST_MODULE_PARAMS(mode, address, port, dataType, numberOfBatches,
                      numberOfTimeSamples, bufferMultiplier);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_TCP_RECEIVER_MODULE_HH
