#ifndef JETSTREAM_DOMAINS_IO_UDP_RECEIVER_MODULE_HH
#define JETSTREAM_DOMAINS_IO_UDP_RECEIVER_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct UdpReceiver : public Module::Config {
    std::string address = "0.0.0.0";
    U64 port = 5000;
    std::string networkInterface = "auto";
    bool filterSource = false;
    std::string sourceAddress = "";
    std::string dataType = "CF32";
    U64 numberOfBatches = 8;
    U64 numberOfTimeSamples = 8192;
    U64 bufferMultiplier = 4;

    JST_MODULE_TYPE(udp_receiver);
    JST_MODULE_PARAMS(address, port, networkInterface, filterSource,
                      sourceAddress, dataType, numberOfBatches,
                      numberOfTimeSamples, bufferMultiplier);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_UDP_RECEIVER_MODULE_HH
