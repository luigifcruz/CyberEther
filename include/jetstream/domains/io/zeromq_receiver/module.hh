#ifndef JETSTREAM_DOMAINS_IO_ZEROMQ_RECEIVER_MODULE_HH
#define JETSTREAM_DOMAINS_IO_ZEROMQ_RECEIVER_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct ZeromqReceiver : public Module::Config {
    std::string pattern = "subscribe";
    std::string mode = "connect";
    std::string address = "127.0.0.1";
    U64 port = 5555;
    std::string dataType = "CF32";
    U64 numberOfBatches = 8;
    U64 numberOfTimeSamples = 8192;
    U64 bufferMultiplier = 4;

    JST_MODULE_TYPE(zeromq_receiver);
    JST_MODULE_PARAMS(pattern, mode, address, port, dataType, numberOfBatches,
                      numberOfTimeSamples, bufferMultiplier);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_ZEROMQ_RECEIVER_MODULE_HH
